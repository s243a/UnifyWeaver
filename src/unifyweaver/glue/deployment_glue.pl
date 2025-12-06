/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Deployment Glue - Automatic deployment and lifecycle management for remote services
 *
 * This module provides:
 * - SSH-based deployment with agent forwarding
 * - Service lifecycle management (start/stop/restart)
 * - Change detection and automatic redeployment
 * - Security validation (encryption requirements)
 * - Generated deployment scripts
 */

:- module(deployment_glue, [
    % Service declarations
    declare_service/2,              % declare_service(+Name, +Options)
    service_config/2,               % service_config(?Name, ?Options)
    undeclare_service/1,            % undeclare_service(+Name)

    % Deployment method declarations
    declare_deploy_method/3,        % declare_deploy_method(+Service, +Method, +Options)
    deploy_method_config/3,         % deploy_method_config(?Service, ?Method, ?Options)

    % Source tracking
    declare_service_sources/2,      % declare_service_sources(+Service, +SourcePatterns)
    service_sources/2,              % service_sources(?Service, ?Sources)

    % Change detection
    compute_source_hash/2,          % compute_source_hash(+Service, -Hash)
    check_for_changes/2,            % check_for_changes(+Service, -Changes)
    store_deployed_hash/2,          % store_deployed_hash(+Service, +Hash)
    deployed_hash/2,                % deployed_hash(?Service, ?Hash)

    % Lifecycle hooks
    declare_lifecycle_hook/3,       % declare_lifecycle_hook(+Service, +Event, +Action)
    lifecycle_hooks/2,              % lifecycle_hooks(?Service, ?Hooks)

    % Security validation
    validate_security/2,            % validate_security(+Service, -Errors)
    requires_encryption/1,          % requires_encryption(+Service)
    is_local_service/1,             % is_local_service(+Service)

    % Deployment script generation
    generate_deploy_script/3,       % generate_deploy_script(+Service, +Options, -Script)
    generate_ssh_deploy/3,          % generate_ssh_deploy(+Service, +Options, -Script)
    generate_systemd_unit/3,        % generate_systemd_unit(+Service, +Options, -Unit)
    generate_health_check_script/3, % generate_health_check_script(+Service, +Options, -Script)

    % Deployment operations
    deploy_service/2,               % deploy_service(+Service, -Result)
    service_status/2,               % service_status(+Service, -Status)

    % Lifecycle operations
    start_service/2,                % start_service(+Service, -Result)
    stop_service/2,                 % stop_service(+Service, -Result)
    restart_service/2               % restart_service(+Service, -Result)
]).

:- use_module(library(lists)).
:- use_module(library(process)).
:- use_module(library(filesex)).
:- use_module(library(sha)).

%% ============================================
%% Dynamic Storage
%% ============================================

:- dynamic service_db/2.            % service_db(Name, Options)
:- dynamic deploy_method_db/3.      % deploy_method_db(Service, Method, Options)
:- dynamic service_sources_db/2.    % service_sources_db(Service, Sources)
:- dynamic deployed_hash_db/2.      % deployed_hash_db(Service, Hash)
:- dynamic lifecycle_hook_db/3.     % lifecycle_hook_db(Service, Event, Action)

%% ============================================
%% Service Declarations
%% ============================================

%% declare_service(+Name, +Options)
%  Declare a remote service with its configuration.
%
%  Options:
%    - host(Host)           : Remote hostname or IP
%    - port(Port)           : Service port (default: 8080)
%    - target(Target)       : Compilation target (python, go, rust, etc.)
%    - entry_point(File)    : Main file to execute
%    - lifecycle(Type)      : persistent | transient | on_demand | pipeline_bound
%    - transport(T)         : http | https | ssh (default: https for remote)
%    - idle_timeout(Secs)   : For transient services
%    - max_lifetime(Secs)   : Maximum runtime
%
declare_service(Name, Options) :-
    atom(Name),
    is_list(Options),
    retractall(service_db(Name, _)),
    assertz(service_db(Name, Options)).

%% service_config(?Name, ?Options)
%  Query service configuration.
%
service_config(Name, Options) :-
    service_db(Name, Options).

%% undeclare_service(+Name)
%  Remove a service declaration.
%
undeclare_service(Name) :-
    retractall(service_db(Name, _)),
    retractall(deploy_method_db(Name, _, _)),
    retractall(service_sources_db(Name, _)),
    retractall(deployed_hash_db(Name, _)),
    retractall(lifecycle_hook_db(Name, _, _)).

%% ============================================
%% Deployment Method Declarations
%% ============================================

%% declare_deploy_method(+Service, +Method, +Options)
%  Declare deployment method for a service.
%
%  Method: ssh | docker | local
%
%  SSH Options:
%    - host(Host)           : Remote host
%    - user(User)           : SSH user
%    - agent(Bool)          : Use SSH agent (default: true)
%    - key_file(Path)       : Explicit key file
%    - remote_dir(Path)     : Deployment directory on remote
%
declare_deploy_method(Service, Method, Options) :-
    atom(Service),
    atom(Method),
    is_list(Options),
    retractall(deploy_method_db(Service, _, _)),
    assertz(deploy_method_db(Service, Method, Options)).

%% deploy_method_config(?Service, ?Method, ?Options)
%  Query deployment method configuration.
%
deploy_method_config(Service, Method, Options) :-
    deploy_method_db(Service, Method, Options).

%% ============================================
%% Source Tracking
%% ============================================

%% declare_service_sources(+Service, +SourcePatterns)
%  Declare source files/patterns for change detection.
%
%  SourcePatterns: List of file paths or glob patterns
%    e.g., ['src/ml/**/*.py', 'models/*.pkl', 'requirements.txt']
%
declare_service_sources(Service, SourcePatterns) :-
    atom(Service),
    is_list(SourcePatterns),
    retractall(service_sources_db(Service, _)),
    assertz(service_sources_db(Service, SourcePatterns)).

%% service_sources(?Service, ?Sources)
%  Query service source patterns.
%
service_sources(Service, Sources) :-
    service_sources_db(Service, Sources).

%% ============================================
%% Change Detection
%% ============================================

%% compute_source_hash(+Service, -Hash)
%  Compute SHA256 hash of all source files for a service.
%
compute_source_hash(Service, Hash) :-
    service_sources_db(Service, Patterns),
    expand_source_patterns(Patterns, Files),
    sort(Files, SortedFiles),  % Deterministic order
    compute_files_hash(SortedFiles, Hash).

%% expand_source_patterns(+Patterns, -Files)
%  Expand glob patterns to actual file paths.
%
expand_source_patterns([], []).
expand_source_patterns([Pattern|Rest], AllFiles) :-
    expand_file_name(Pattern, MatchedFiles),
    include(exists_file, MatchedFiles, ExistingFiles),
    expand_source_patterns(Rest, RestFiles),
    append(ExistingFiles, RestFiles, AllFiles).

%% compute_files_hash(+Files, -Hash)
%  Compute combined hash of multiple files.
%
compute_files_hash(Files, Hash) :-
    maplist(file_content_for_hash, Files, Contents),
    atomic_list_concat(Contents, Combined),
    sha_hash(Combined, HashBytes, [algorithm(sha256)]),
    hash_bytes_to_hex(HashBytes, Hash).

%% file_content_for_hash(+File, -Content)
%  Read file content for hashing, including filename for uniqueness.
%
file_content_for_hash(File, Content) :-
    (   exists_file(File)
    ->  read_file_to_string(File, FileContent, []),
        format(atom(Content), '~w:~w', [File, FileContent])
    ;   format(atom(Content), '~w:missing', [File])
    ).

%% hash_bytes_to_hex(+Bytes, -Hex)
%  Convert hash bytes to hexadecimal string.
%
hash_bytes_to_hex(Bytes, Hex) :-
    maplist(byte_to_hex, Bytes, HexParts),
    atomic_list_concat(HexParts, Hex).

byte_to_hex(Byte, Hex) :-
    format(atom(Hex), '~`0t~16r~2|', [Byte]).

%% check_for_changes(+Service, -Changes)
%  Check if service sources have changed since last deployment.
%
%  Changes = no_changes | changed(OldHash, NewHash) | never_deployed
%
check_for_changes(Service, Changes) :-
    compute_source_hash(Service, CurrentHash),
    (   deployed_hash_db(Service, DeployedHash)
    ->  (   CurrentHash == DeployedHash
        ->  Changes = no_changes
        ;   Changes = changed(DeployedHash, CurrentHash)
        )
    ;   Changes = never_deployed
    ).

%% store_deployed_hash(+Service, +Hash)
%  Store the hash of deployed sources.
%
store_deployed_hash(Service, Hash) :-
    retractall(deployed_hash_db(Service, _)),
    assertz(deployed_hash_db(Service, Hash)).

%% deployed_hash(?Service, ?Hash)
%  Query deployed hash.
%
deployed_hash(Service, Hash) :-
    deployed_hash_db(Service, Hash).

%% ============================================
%% Lifecycle Hooks
%% ============================================

%% declare_lifecycle_hook(+Service, +Event, +Action)
%  Declare a lifecycle hook for a service.
%
%  Events:
%    - pre_shutdown       : Before stopping service
%    - post_shutdown      : After stopping service
%    - pre_deploy         : Before deployment
%    - post_deploy        : After deployment
%    - on_health_failure  : When health check fails
%
%  Action: Atom or compound term describing the action
%    - drain_connections
%    - save_state
%    - health_check
%    - warm_cache
%    - custom(Command)
%
declare_lifecycle_hook(Service, Event, Action) :-
    atom(Service),
    atom(Event),
    assertz(lifecycle_hook_db(Service, Event, Action)).

%% lifecycle_hooks(?Service, ?Hooks)
%  Query all lifecycle hooks for a service.
%
lifecycle_hooks(Service, Hooks) :-
    findall(hook(Event, Action), lifecycle_hook_db(Service, Event, Action), Hooks).

%% ============================================
%% Security Validation
%% ============================================

%% validate_security(+Service, -Errors)
%  Validate security requirements for a service.
%  Returns empty list if valid, otherwise list of error terms.
%
validate_security(Service, Errors) :-
    service_config(Service, Options),
    findall(Error, security_violation(Service, Options, Error), Errors).

%% security_violation(+Service, +Options, -Error)
%  Check for specific security violations.
%
security_violation(_Service, Options, Error) :-
    member(host(Host), Options),
    \+ is_localhost(Host),
    member(transport(http), Options),
    Error = remote_requires_encryption(Host).

security_violation(_Service, Options, Error) :-
    member(host(Host), Options),
    \+ is_localhost(Host),
    \+ member(transport(_), Options),
    Error = remote_missing_transport(Host).

%% is_localhost(+Host)
%  Check if host is localhost.
%
is_localhost(localhost).
is_localhost('127.0.0.1').
is_localhost('::1').

%% requires_encryption(+Service)
%  Check if service requires encryption.
%
requires_encryption(Service) :-
    service_config(Service, Options),
    member(host(Host), Options),
    \+ is_localhost(Host).

%% is_local_service(+Service)
%  Check if service is local (doesn't require encryption).
%
is_local_service(Service) :-
    service_config(Service, Options),
    (   \+ member(host(_), Options)
    ;   member(host(Host), Options),
        is_localhost(Host)
    ).

%% ============================================
%% Deployment Script Generation
%% ============================================

%% generate_deploy_script(+Service, +Options, -Script)
%  Generate deployment script based on configured method.
%
generate_deploy_script(Service, Options, Script) :-
    deploy_method_config(Service, Method, MethodOptions),
    merge_options(Options, MethodOptions, MergedOptions),
    generate_deploy_script_for_method(Method, Service, MergedOptions, Script).

generate_deploy_script_for_method(ssh, Service, Options, Script) :-
    generate_ssh_deploy(Service, Options, Script).
generate_deploy_script_for_method(local, Service, Options, Script) :-
    generate_local_deploy(Service, Options, Script).

%% generate_ssh_deploy(+Service, +Options, -Script)
%  Generate SSH deployment script.
%
generate_ssh_deploy(Service, Options, Script) :-
    service_config(Service, ServiceOptions),

    % Extract required options (check Options first, then ServiceOptions, then default)
    option_or_default(host, Options, ServiceOptions, 'localhost', Host),
    option_or_default(user, Options, ServiceOptions, 'deploy', User),
    option_or_default(remote_dir, Options, ServiceOptions, '/opt/unifyweaver/services', BaseRemoteDir),
    option_or_default(port, Options, ServiceOptions, 8080, Port),
    option_or_default(target, Options, ServiceOptions, python, Target),
    option_or_default(entry_point, Options, ServiceOptions, 'server.py', EntryPoint),

    % Build remote directory path
    format(atom(RemoteDir), '~w/~w', [BaseRemoteDir, Service]),

    % Get source patterns for rsync
    (   service_sources(Service, Sources)
    ->  true
    ;   Sources = ['.']
    ),

    % Get lifecycle hooks
    lifecycle_hooks(Service, Hooks),

    % Generate script
    generate_ssh_script_content(Service, Host, User, RemoteDir, Port, Target,
                                 EntryPoint, Sources, Hooks, Script).

%% generate_ssh_script_content/10
%  Generate the actual SSH deployment script content.
%
generate_ssh_script_content(Service, Host, User, RemoteDir, Port, Target,
                            EntryPoint, _Sources, Hooks, Script) :-
    % Pre-shutdown hooks
    generate_hook_commands(Hooks, pre_shutdown, Host, User, PreShutdownCmds),

    % Post-deploy hooks
    generate_hook_commands(Hooks, post_deploy, Host, User, PostDeployCmds),

    % Generate startup command based on target
    generate_startup_command(Target, EntryPoint, Port, StartupCmd),

    format(atom(Script), '#!/bin/bash
# Deployment script for ~w
# Generated by UnifyWeaver deployment_glue

set -euo pipefail

SERVICE="~w"
HOST="~w"
USER="~w"
REMOTE_DIR="~w"
PORT="~w"

echo "=== Deploying ${SERVICE} to ${HOST} ==="

# Check SSH agent
if ! ssh-add -l &>/dev/null; then
    echo "Warning: SSH agent not running or no keys loaded"
    echo "Attempting connection anyway..."
fi

# Check connectivity
echo "Checking connectivity..."
if ! ssh -o ConnectTimeout=10 "${USER}@${HOST}" "echo ok" >/dev/null 2>&1; then
    echo "Error: Cannot connect to ${HOST}"
    exit 1
fi

~w

# Create remote directory
echo "Creating remote directory..."
ssh "${USER}@${HOST}" "mkdir -p ${REMOTE_DIR}"

# Sync sources
echo "Syncing sources..."
rsync -avz --delete \\
    --exclude "*.pyc" \\
    --exclude "__pycache__" \\
    --exclude ".git" \\
    --exclude "*.log" \\
    ./ "${USER}@${HOST}:${REMOTE_DIR}/"

# Install dependencies if requirements.txt exists
echo "Installing dependencies..."
ssh "${USER}@${HOST}" "cd ${REMOTE_DIR} && \\
    if [ -f requirements.txt ]; then \\
        pip install -r requirements.txt; \\
    elif [ -f go.mod ]; then \\
        go mod download; \\
    elif [ -f Cargo.toml ]; then \\
        cargo build --release; \\
    fi"

# Restart service
echo "Restarting service..."
ssh "${USER}@${HOST}" "cd ${REMOTE_DIR} && \\
    pkill -f \\"~w\\" || true && \\
    nohup ~w > service.log 2>&1 &"

~w

echo "=== Deployment complete ==="
echo "Service ~w running on ${HOST}:${PORT}"
', [Service, Service, Host, User, RemoteDir, Port,
    PreShutdownCmds, EntryPoint, StartupCmd, PostDeployCmds, Service]).

%% generate_hook_commands(+Hooks, +Event, +Host, +User, -Commands)
%  Generate shell commands for lifecycle hooks.
%
generate_hook_commands(Hooks, Event, Host, User, Commands) :-
    findall(Cmd, (
        member(hook(Event, Action), Hooks),
        hook_to_command(Action, Host, User, Cmd)
    ), CmdList),
    (   CmdList == []
    ->  Commands = ''
    ;   atomic_list_concat(CmdList, '\n', Commands)
    ).

%% hook_to_command(+Action, +Host, +User, -Command)
%  Convert a lifecycle action to a shell command.
%
hook_to_command(drain_connections, Host, User, Cmd) :-
    format(atom(Cmd), 'echo "Draining connections..."\nssh "~w@~w" "sleep 5"  # Allow connections to drain', [User, Host]).
hook_to_command(health_check, Host, _User, Cmd) :-
    format(atom(Cmd), 'echo "Running health check..."\nfor i in {1..30}; do\n    if curl -sf "http://~w:${PORT}/health" >/dev/null; then\n        echo "Health check passed"\n        break\n    fi\n    sleep 1\ndone', [Host]).
hook_to_command(warm_cache, Host, User, Cmd) :-
    format(atom(Cmd), 'echo "Warming cache..."\nssh "~w@~w" "curl -s http://localhost:${PORT}/warmup || true"', [User, Host]).
hook_to_command(save_state, Host, User, Cmd) :-
    format(atom(Cmd), 'echo "Saving state..."\nssh "~w@~w" "cd ${REMOTE_DIR} && ./save_state.sh || true"', [User, Host]).
hook_to_command(custom(Command), Host, User, Cmd) :-
    format(atom(Cmd), 'echo "Running custom hook..."\nssh "~w@~w" "~w"', [User, Host, Command]).

%% generate_startup_command(+Target, +EntryPoint, +Port, -Command)
%  Generate startup command based on target language.
%
generate_startup_command(python, EntryPoint, Port, Cmd) :-
    format(atom(Cmd), 'python3 ~w --port ~w', [EntryPoint, Port]).
generate_startup_command(go, EntryPoint, Port, Cmd) :-
    format(atom(Cmd), './~w -port ~w', [EntryPoint, Port]).
generate_startup_command(rust, EntryPoint, Port, Cmd) :-
    format(atom(Cmd), './target/release/~w --port ~w', [EntryPoint, Port]).
generate_startup_command(node, EntryPoint, Port, Cmd) :-
    format(atom(Cmd), 'node ~w --port ~w', [EntryPoint, Port]).
generate_startup_command(_, EntryPoint, Port, Cmd) :-
    format(atom(Cmd), './~w --port ~w', [EntryPoint, Port]).

%% generate_local_deploy(+Service, +Options, -Script)
%  Generate local deployment script (no SSH).
%
generate_local_deploy(Service, _Options, Script) :-
    service_config(Service, ServiceOptions),
    option_or_default(port, ServiceOptions, 8080, Port),
    option_or_default(target, ServiceOptions, python, Target),
    option_or_default(entry_point, ServiceOptions, 'server.py', EntryPoint),

    generate_startup_command(Target, EntryPoint, Port, StartupCmd),

    format(atom(Script), '#!/bin/bash
# Local deployment script for ~w
# Generated by UnifyWeaver deployment_glue

set -euo pipefail

SERVICE="~w"
PORT="~w"

echo "=== Starting ${SERVICE} locally ==="

# Install dependencies
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
elif [ -f go.mod ]; then
    go build .
elif [ -f Cargo.toml ]; then
    cargo build --release
fi

# Stop existing instance
pkill -f "~w" || true

# Start service
nohup ~w > service.log 2>&1 &

echo "Service ~w running on localhost:${PORT}"
', [Service, Service, Port, EntryPoint, StartupCmd, Service]).

%% ============================================
%% Systemd Unit Generation
%% ============================================

%% generate_systemd_unit(+Service, +Options, -Unit)
%  Generate systemd service unit file.
%
generate_systemd_unit(Service, Options, Unit) :-
    service_config(Service, ServiceOptions),

    option_or_default(remote_dir, Options, '/opt/unifyweaver/services', BaseDir),
    option_or_default(user, Options, 'deploy', User),
    option_or_default(target, ServiceOptions, python, Target),
    option_or_default(entry_point, ServiceOptions, 'server.py', EntryPoint),
    option_or_default(port, ServiceOptions, 8080, Port),

    format(atom(WorkDir), '~w/~w', [BaseDir, Service]),
    generate_startup_command(Target, EntryPoint, Port, ExecStart),

    format(atom(Unit), '[Unit]
Description=UnifyWeaver Service: ~w
After=network.target

[Service]
Type=simple
User=~w
WorkingDirectory=~w
ExecStart=~w
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
', [Service, User, WorkDir, ExecStart]).

%% ============================================
%% Health Check Script Generation
%% ============================================

%% generate_health_check_script(+Service, +Options, -Script)
%  Generate health check script.
%
generate_health_check_script(Service, Options, Script) :-
    service_config(Service, ServiceOptions),

    option_or_default(host, Options, ServiceOptions, 'localhost', Host),
    option_or_default(port, ServiceOptions, 8080, Port),
    option_or_default(health_endpoint, Options, '/health', Endpoint),
    option_or_default(timeout, Options, 5, Timeout),
    option_or_default(retries, Options, 3, Retries),

    % Determine protocol
    (   is_localhost(Host)
    ->  Protocol = 'http'
    ;   Protocol = 'https'
    ),

    format(atom(Script), '#!/bin/bash
# Health check script for ~w
# Generated by UnifyWeaver deployment_glue

HOST="~w"
PORT="~w"
ENDPOINT="~w"
TIMEOUT="~w"
RETRIES="~w"
PROTOCOL="~w"

URL="${PROTOCOL}://${HOST}:${PORT}${ENDPOINT}"

for i in $(seq 1 $RETRIES); do
    if curl -sf --max-time $TIMEOUT "$URL" >/dev/null; then
        echo "OK: ~w is healthy"
        exit 0
    fi
    echo "Attempt $i/$RETRIES failed, retrying..."
    sleep 2
done

echo "CRITICAL: ~w health check failed after $RETRIES attempts"
exit 2
', [Service, Host, Port, Endpoint, Timeout, Retries, Protocol, Service, Service]).

%% ============================================
%% Deployment Operations
%% ============================================

%% deploy_service(+Service, -Result)
%  Execute deployment for a service.
%
deploy_service(Service, Result) :-
    % Validate security first
    validate_security(Service, SecurityErrors),
    (   SecurityErrors \== []
    ->  Result = error(security_validation_failed(SecurityErrors))
    ;   % Check for changes
        check_for_changes(Service, Changes),
        (   Changes == no_changes
        ->  Result = unchanged
        ;   % Generate and execute deployment
            generate_deploy_script(Service, [], Script),
            execute_deployment(Script, ExecResult),
            (   ExecResult == success
            ->  % Store new hash
                compute_source_hash(Service, NewHash),
                store_deployed_hash(Service, NewHash),
                Result = deployed(Changes)
            ;   Result = error(deployment_failed(ExecResult))
            )
        )
    ).

%% execute_deployment(+Script, -Result)
%  Execute a deployment script.
%
execute_deployment(Script, Result) :-
    % Write script to temp file
    tmp_file_stream(text, TmpFile, Stream),
    write(Stream, Script),
    close(Stream),

    % Make executable and run
    process_create(path(chmod), ['+x', TmpFile], []),
    catch(
        (   process_create(path(bash), [TmpFile],
                [stdout(pipe(Out)), stderr(pipe(Err))]),
            read_string(Out, _, _OutStr),
            read_string(Err, _, _ErrStr),
            close(Out),
            close(Err),
            Result = success
        ),
        Error,
        Result = error(Error)
    ),

    % Cleanup
    delete_file(TmpFile).

%% service_status(+Service, -Status)
%  Check status of a deployed service.
%
service_status(Service, Status) :-
    service_config(Service, Options),
    (   member(host(Host), Options)
    ->  true
    ;   Host = localhost
    ),
    (   member(port(Port), Options)
    ->  true
    ;   Port = 8080
    ),

    % Determine protocol
    (   is_localhost(Host)
    ->  Protocol = 'http'
    ;   Protocol = 'https'
    ),

    format(atom(URL), '~w://~w:~w/health', [Protocol, Host, Port]),

    catch(
        (   http_open(URL, Stream, [timeout(5)]),
            close(Stream),
            Status = running
        ),
        _,
        Status = stopped
    ).

%% ============================================
%% Lifecycle Operations
%% ============================================

%% start_service(+Service, -Result)
%  Start a service.
%
start_service(Service, Result) :-
    deploy_method_config(Service, Method, Options),
    start_service_method(Method, Service, Options, Result).

start_service_method(ssh, Service, Options, Result) :-
    service_config(Service, ServiceOptions),
    option_or_default(host, Options, ServiceOptions, 'localhost', Host),
    option_or_default(user, Options, ServiceOptions, 'deploy', User),
    format(atom(Cmd), 'ssh ~w@~w "systemctl --user start ~w"', [User, Host, Service]),
    shell(Cmd, ExitCode),
    (   ExitCode == 0
    ->  Result = started
    ;   Result = error(start_failed(ExitCode))
    ).

start_service_method(local, Service, _Options, Result) :-
    format(atom(Cmd), 'systemctl --user start ~w', [Service]),
    shell(Cmd, ExitCode),
    (   ExitCode == 0
    ->  Result = started
    ;   Result = error(start_failed(ExitCode))
    ).

%% stop_service(+Service, -Result)
%  Stop a service.
%
stop_service(Service, Result) :-
    deploy_method_config(Service, Method, Options),
    stop_service_method(Method, Service, Options, Result).

stop_service_method(ssh, Service, Options, Result) :-
    service_config(Service, ServiceOptions),
    option_or_default(host, Options, ServiceOptions, 'localhost', Host),
    option_or_default(user, Options, ServiceOptions, 'deploy', User),

    % Execute pre-shutdown hooks
    lifecycle_hooks(Service, Hooks),
    generate_hook_commands(Hooks, pre_shutdown, Host, User, PreCmds),
    (   PreCmds \== ''
    ->  format(atom(PreCmd), 'ssh ~w@~w "~w"', [User, Host, PreCmds]),
        shell(PreCmd, _)
    ;   true
    ),

    format(atom(Cmd), 'ssh ~w@~w "systemctl --user stop ~w"', [User, Host, Service]),
    shell(Cmd, ExitCode),
    (   ExitCode == 0
    ->  Result = stopped
    ;   Result = error(stop_failed(ExitCode))
    ).

stop_service_method(local, Service, _Options, Result) :-
    format(atom(Cmd), 'systemctl --user stop ~w', [Service]),
    shell(Cmd, ExitCode),
    (   ExitCode == 0
    ->  Result = stopped
    ;   Result = error(stop_failed(ExitCode))
    ).

%% restart_service(+Service, -Result)
%  Restart a service.
%
restart_service(Service, Result) :-
    stop_service(Service, StopResult),
    (   StopResult = stopped
    ->  start_service(Service, Result)
    ;   StopResult = error(_)
    ->  % Try to start anyway
        start_service(Service, Result)
    ;   Result = StopResult
    ).

%% ============================================
%% Helper Predicates
%% ============================================

%% option_or_default(+Key, +Options1, +Options2, +Default, -Value)
%  Get option value from Options1, then Options2, then Default.
%
option_or_default(Key, Options1, Options2, Default, Value) :-
    Term =.. [Key, Value],
    (   member(Term, Options1)
    ->  true
    ;   member(Term, Options2)
    ->  true
    ;   Value = Default
    ).

option_or_default(Key, Options, Default, Value) :-
    Term =.. [Key, Value],
    (   member(Term, Options)
    ->  true
    ;   Value = Default
    ).

%% merge_options(+Options1, +Options2, -Merged)
%  Merge two option lists, Options1 takes precedence.
%
merge_options(Options1, Options2, Merged) :-
    append(Options1, Options2, Merged).
