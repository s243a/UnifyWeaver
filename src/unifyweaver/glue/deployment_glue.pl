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
    restart_service/2,              % restart_service(+Service, -Result)

    % Phase 6b: Advanced deployment
    % Multi-host support
    declare_service_hosts/2,        % declare_service_hosts(+Service, +Hosts)
    service_hosts/2,                % service_hosts(?Service, ?Hosts)
    deploy_to_all_hosts/2,          % deploy_to_all_hosts(+Service, -Results)

    % Rollback support
    store_rollback_hash/2,          % store_rollback_hash(+Service, +Hash)
    rollback_hash/2,                % rollback_hash(?Service, ?Hash)
    rollback_service/2,             % rollback_service(+Service, -Result)
    deploy_with_rollback/2,         % deploy_with_rollback(+Service, -Result)

    % Graceful shutdown
    graceful_stop/3,                % graceful_stop(+Service, +Options, -Result)
    drain_connections/3,            % drain_connections(+Service, +Options, -Result)

    % Health check integration
    run_health_check/3,             % run_health_check(+Service, +Options, -Result)
    wait_for_healthy/3,             % wait_for_healthy(+Service, +Options, -Result)

    % Deployment with full lifecycle
    deploy_with_hooks/2,            % deploy_with_hooks(+Service, -Result)
    generate_rollback_script/3,     % generate_rollback_script(+Service, +Options, -Script)

    % Hook execution (internal but exported for testing)
    execute_hooks/3,                % execute_hooks(+Service, +Event, -Result)

    % Phase 6c: Error Handling
    % Retry policies
    declare_retry_policy/2,         % declare_retry_policy(+Service, +Policy)
    retry_policy/2,                 % retry_policy(?Service, ?Policy)
    call_with_retry/4,              % call_with_retry(+Service, +Operation, +Args, -Result)

    % Fallback mechanisms
    declare_fallback/2,             % declare_fallback(+Service, +Fallback)
    fallback_config/2,              % fallback_config(?Service, ?Fallback)
    call_with_fallback/4,           % call_with_fallback(+Service, +Operation, +Args, -Result)

    % Circuit breaker
    declare_circuit_breaker/2,      % declare_circuit_breaker(+Service, +Config)
    circuit_breaker_config/2,       % circuit_breaker_config(?Service, ?Config)
    circuit_state/2,                % circuit_state(?Service, ?State)
    call_with_circuit_breaker/4,    % call_with_circuit_breaker(+Service, +Operation, +Args, -Result)
    reset_circuit_breaker/1,        % reset_circuit_breaker(+Service)

    % Timeout configuration
    declare_timeouts/2,             % declare_timeouts(+Service, +Timeouts)
    timeout_config/2,               % timeout_config(?Service, ?Timeouts)
    call_with_timeout/4,            % call_with_timeout(+Service, +Operation, +Args, -Result)

    % Combined error handling
    protected_call/4,               % protected_call(+Service, +Operation, +Args, -Result)

    % Internal (exported for testing)
    record_circuit_failure/1,       % record_circuit_failure(+Service)
    record_circuit_success/1,       % record_circuit_success(+Service)

    % Phase 6d: Monitoring
    % Health check monitoring
    declare_health_check/2,         % declare_health_check(+Service, +Config)
    health_check_config/2,          % health_check_config(?Service, ?Config)
    start_health_monitor/2,         % start_health_monitor(+Service, -Result)
    stop_health_monitor/1,          % stop_health_monitor(+Service)
    health_status/2,                % health_status(?Service, ?Status)

    % Metrics collection
    declare_metrics/2,              % declare_metrics(+Service, +Config)
    metrics_config/2,               % metrics_config(?Service, ?Config)
    record_metric/3,                % record_metric(+Service, +Metric, +Value)
    get_metrics/2,                  % get_metrics(+Service, -Metrics)
    generate_prometheus_metrics/2,  % generate_prometheus_metrics(+Service, -Output)

    % Structured logging
    declare_logging/2,              % declare_logging(+Service, +Config)
    logging_config/2,               % logging_config(?Service, ?Config)
    log_event/4,                    % log_event(+Service, +Level, +Message, +Data)
    get_log_entries/3,              % get_log_entries(+Service, +Options, -Entries)

    % Alerting
    declare_alert/3,                % declare_alert(+Service, +AlertName, +Config)
    alert_config/3,                 % alert_config(?Service, ?AlertName, ?Config)
    check_alerts/2,                 % check_alerts(+Service, -TriggeredAlerts)
    trigger_alert/3,                % trigger_alert(+Service, +AlertName, +Data)
    alert_history/3                 % alert_history(+Service, +Options, -History)
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

% Phase 6b dynamic storage
:- dynamic service_hosts_db/2.      % service_hosts_db(Service, Hosts)
:- dynamic rollback_hash_db/2.      % rollback_hash_db(Service, Hash)

% Phase 6c dynamic storage
:- dynamic retry_policy_db/2.       % retry_policy_db(Service, Policy)
:- dynamic fallback_config_db/2.    % fallback_config_db(Service, Fallback)
:- dynamic circuit_breaker_db/2.    % circuit_breaker_db(Service, Config)
:- dynamic circuit_state_db/3.      % circuit_state_db(Service, State, Data)
:- dynamic timeout_config_db/2.     % timeout_config_db(Service, Timeouts)

% Phase 6d dynamic storage
:- dynamic health_check_config_db/2.    % health_check_config_db(Service, Config)
:- dynamic health_status_db/3.          % health_status_db(Service, Status, Timestamp)
:- dynamic health_monitor_pid_db/2.     % health_monitor_pid_db(Service, Pid)
:- dynamic metrics_config_db/2.         % metrics_config_db(Service, Config)
:- dynamic metric_data_db/4.            % metric_data_db(Service, Metric, Value, Timestamp)
:- dynamic logging_config_db/2.         % logging_config_db(Service, Config)
:- dynamic log_entry_db/5.              % log_entry_db(Service, Level, Message, Data, Timestamp)
:- dynamic alert_config_db/3.           % alert_config_db(Service, AlertName, Config)
:- dynamic alert_state_db/4.            % alert_state_db(Service, AlertName, State, Since)
:- dynamic alert_history_db/5.          % alert_history_db(Service, AlertName, State, Data, Timestamp)

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

%% ============================================
%% Phase 6b: Multi-Host Support
%% ============================================

%% declare_service_hosts(+Service, +Hosts)
%  Declare multiple hosts for a service (for load balancing/redundancy).
%
%  Hosts: List of host configurations
%    e.g., [host_config('host1.example.com', [user('deploy')]),
%           host_config('host2.example.com', [user('deploy')])]
%
declare_service_hosts(Service, Hosts) :-
    atom(Service),
    is_list(Hosts),
    retractall(service_hosts_db(Service, _)),
    assertz(service_hosts_db(Service, Hosts)).

%% service_hosts(?Service, ?Hosts)
%  Query service hosts.
%
service_hosts(Service, Hosts) :-
    service_hosts_db(Service, Hosts).

%% deploy_to_all_hosts(+Service, -Results)
%  Deploy service to all configured hosts.
%  Returns list of result(Host, Status) terms.
%
deploy_to_all_hosts(Service, Results) :-
    (   service_hosts(Service, Hosts)
    ->  maplist(deploy_to_host(Service), Hosts, Results)
    ;   % Single host from service config
        service_config(Service, Options),
        (   member(host(Host), Options)
        ->  deploy_service(Service, Result),
            Results = [result(Host, Result)]
        ;   Results = [result(localhost, error(no_host_configured))]
        )
    ).

%% deploy_to_host(+Service, +HostConfig, -Result)
%  Deploy to a specific host.
%
deploy_to_host(Service, host_config(Host, HostOptions), result(Host, Result)) :-
    % Temporarily override host in service config
    service_config(Service, OriginalOptions),
    merge_options([host(Host)|HostOptions], OriginalOptions, MergedOptions),

    % Deploy with merged options
    (   catch(
            deploy_with_options(Service, MergedOptions, DeployResult),
            Error,
            DeployResult = error(Error)
        )
    ->  Result = DeployResult
    ;   Result = error(deployment_failed)
    ).

%% deploy_with_options(+Service, +Options, -Result)
%  Deploy service with specific options (internal helper).
%
deploy_with_options(Service, Options, Result) :-
    validate_security_with_options(Options, SecurityErrors),
    (   SecurityErrors \== []
    ->  Result = error(security_validation_failed(SecurityErrors))
    ;   generate_deploy_script(Service, Options, Script),
        execute_deployment(Script, ExecResult),
        (   ExecResult == success
        ->  compute_source_hash(Service, NewHash),
            store_deployed_hash(Service, NewHash),
            Result = deployed
        ;   Result = error(deployment_failed(ExecResult))
        )
    ).

%% validate_security_with_options(+Options, -Errors)
%  Validate security for given options.
%
validate_security_with_options(Options, Errors) :-
    findall(Error, security_violation_opts(Options, Error), Errors).

security_violation_opts(Options, remote_requires_encryption(Host)) :-
    member(host(Host), Options),
    \+ is_localhost(Host),
    member(transport(http), Options).

security_violation_opts(Options, remote_missing_transport(Host)) :-
    member(host(Host), Options),
    \+ is_localhost(Host),
    \+ member(transport(_), Options).

%% ============================================
%% Phase 6b: Rollback Support
%% ============================================

%% store_rollback_hash(+Service, +Hash)
%  Store hash for potential rollback (previous good version).
%
store_rollback_hash(Service, Hash) :-
    retractall(rollback_hash_db(Service, _)),
    assertz(rollback_hash_db(Service, Hash)).

%% rollback_hash(?Service, ?Hash)
%  Query rollback hash.
%
rollback_hash(Service, Hash) :-
    rollback_hash_db(Service, Hash).

%% rollback_service(+Service, -Result)
%  Rollback to previous version using stored rollback hash.
%
rollback_service(Service, Result) :-
    (   rollback_hash(Service, RollbackHash)
    ->  % Generate rollback script
        generate_rollback_script(Service, [], Script),
        execute_deployment(Script, ExecResult),
        (   ExecResult == success
        ->  % Update deployed hash to rollback version
            store_deployed_hash(Service, RollbackHash),
            Result = rolled_back(RollbackHash)
        ;   Result = error(rollback_failed(ExecResult))
        )
    ;   Result = error(no_rollback_available)
    ).

%% deploy_with_rollback(+Service, -Result)
%  Deploy with automatic rollback on health check failure.
%
deploy_with_rollback(Service, Result) :-
    % Store current hash as rollback point
    (   deployed_hash(Service, CurrentHash)
    ->  store_rollback_hash(Service, CurrentHash)
    ;   true  % No previous deployment
    ),

    % Attempt deployment
    deploy_with_hooks(Service, DeployResult),

    (   DeployResult = deployed
    ->  % Run health check
        run_health_check(Service, [retries(5), delay(2)], HealthResult),
        (   HealthResult == healthy
        ->  Result = deployed
        ;   % Health check failed - rollback
            format(user_error, 'Health check failed, initiating rollback...~n', []),
            rollback_service(Service, RollbackResult),
            Result = rolled_back_after_failure(RollbackResult)
        )
    ;   Result = DeployResult
    ).

%% generate_rollback_script(+Service, +Options, -Script)
%  Generate script to rollback to previous version.
%
generate_rollback_script(Service, Options, Script) :-
    deploy_method_config(Service, Method, MethodOptions),
    merge_options(Options, MethodOptions, MergedOptions),
    service_config(Service, ServiceOptions),

    option_or_default(host, MergedOptions, ServiceOptions, 'localhost', Host),
    option_or_default(user, MergedOptions, ServiceOptions, 'deploy', User),
    option_or_default(remote_dir, MergedOptions, ServiceOptions, '/opt/unifyweaver/services', BaseRemoteDir),

    format(atom(RemoteDir), '~w/~w', [BaseRemoteDir, Service]),

    (   Method == ssh
    ->  format(atom(Script), '#!/bin/bash
# Rollback script for ~w
# Generated by UnifyWeaver deployment_glue

set -euo pipefail

SERVICE="~w"
HOST="~w"
USER="~w"
REMOTE_DIR="~w"

echo "=== Rolling back ${SERVICE} on ${HOST} ==="

# Check for backup
ssh "${USER}@${HOST}" "
    if [ -d ${REMOTE_DIR}.backup ]; then
        echo \\"Restoring from backup...\\"
        rm -rf ${REMOTE_DIR}
        mv ${REMOTE_DIR}.backup ${REMOTE_DIR}
        echo \\"Backup restored\\"
    else
        echo \\"No backup found at ${REMOTE_DIR}.backup\\"
        exit 1
    fi
"

# Restart service
echo "Restarting service..."
ssh "${USER}@${HOST}" "systemctl --user restart ${SERVICE} || pkill -f ${SERVICE} && cd ${REMOTE_DIR} && nohup ./start.sh > service.log 2>&1 &"

echo "=== Rollback complete ==="
', [Service, Service, Host, User, RemoteDir])
    ;   % Local rollback
        format(atom(Script), '#!/bin/bash
# Local rollback script for ~w
# Generated by UnifyWeaver deployment_glue

set -euo pipefail

SERVICE="~w"
DIR="."

echo "=== Rolling back ${SERVICE} locally ==="

if [ -d "${DIR}.backup" ]; then
    echo "Restoring from backup..."
    rm -rf "${DIR}"
    mv "${DIR}.backup" "${DIR}"
    echo "Backup restored"
else
    echo "No backup found"
    exit 1
fi

echo "=== Rollback complete ==="
', [Service, Service])
    ).

%% ============================================
%% Phase 6b: Graceful Shutdown
%% ============================================

%% graceful_stop(+Service, +Options, -Result)
%  Stop service gracefully with connection draining.
%
%  Options:
%    - drain_timeout(Seconds)  : Time to wait for connections to drain (default: 30)
%    - force_after(Seconds)    : Force kill after this time (default: 60)
%
graceful_stop(Service, Options, Result) :-
    option_or_default(drain_timeout, Options, 30, DrainTimeout),
    option_or_default(force_after, Options, 60, ForceTimeout),

    % Execute pre-shutdown hooks
    execute_hooks(Service, pre_shutdown, PreResult),

    (   PreResult == ok
    ->  % Drain connections
        drain_connections(Service, [timeout(DrainTimeout)], DrainResult),

        (   DrainResult == drained
        ->  % Stop service
            stop_service(Service, StopResult),
            Result = StopResult
        ;   DrainResult == timeout
        ->  % Force stop after timeout
            format(user_error, 'Drain timeout, forcing stop after ~w seconds...~n', [ForceTimeout]),
            force_stop_service(Service, ForceTimeout, Result)
        ;   Result = error(drain_failed(DrainResult))
        )
    ;   Result = error(pre_shutdown_hook_failed(PreResult))
    ).

%% drain_connections(+Service, +Options, -Result)
%  Wait for active connections to complete.
%
drain_connections(Service, Options, Result) :-
    option_or_default(timeout, Options, 30, Timeout),

    deploy_method_config(Service, Method, MethodOptions),
    service_config(Service, ServiceOptions),

    option_or_default(host, MethodOptions, ServiceOptions, 'localhost', Host),
    option_or_default(user, MethodOptions, ServiceOptions, 'deploy', User),
    option_or_default(port, ServiceOptions, 8080, Port),

    (   Method == ssh
    ->  % Remote drain - signal service to stop accepting new connections
        format(atom(DrainCmd), 'ssh ~w@~w "curl -sf http://localhost:~w/drain || true; sleep ~w"',
               [User, Host, Port, Timeout]),
        (   shell(DrainCmd, 0)
        ->  Result = drained
        ;   Result = timeout
        )
    ;   % Local drain
        format(atom(DrainCmd), 'curl -sf http://localhost:~w/drain || true; sleep ~w',
               [Port, Timeout]),
        (   shell(DrainCmd, 0)
        ->  Result = drained
        ;   Result = timeout
        )
    ).

%% force_stop_service(+Service, +Timeout, -Result)
%  Force stop a service after timeout.
%
force_stop_service(Service, _Timeout, Result) :-
    deploy_method_config(Service, Method, MethodOptions),
    service_config(Service, ServiceOptions),

    option_or_default(host, MethodOptions, ServiceOptions, 'localhost', Host),
    option_or_default(user, MethodOptions, ServiceOptions, 'deploy', User),

    (   Method == ssh
    ->  format(atom(Cmd), 'ssh ~w@~w "pkill -9 -f ~w || true"', [User, Host, Service]),
        shell(Cmd, _),
        Result = force_stopped
    ;   format(atom(Cmd), 'pkill -9 -f ~w || true', [Service]),
        shell(Cmd, _),
        Result = force_stopped
    ).

%% execute_hooks(+Service, +Event, -Result)
%  Execute all hooks for an event.
%
execute_hooks(Service, Event, Result) :-
    lifecycle_hooks(Service, Hooks),
    findall(Action, member(hook(Event, Action), Hooks), Actions),
    execute_hook_actions(Service, Actions, Result).

execute_hook_actions(_Service, [], ok).
execute_hook_actions(Service, [Action|Rest], Result) :-
    execute_single_hook(Service, Action, ActionResult),
    (   ActionResult == ok
    ->  execute_hook_actions(Service, Rest, Result)
    ;   Result = error(hook_failed(Action, ActionResult))
    ).

%% execute_single_hook(+Service, +Action, -Result)
%  Execute a single hook action.
%
execute_single_hook(Service, drain_connections, Result) :-
    drain_connections(Service, [timeout(30)], DrainResult),
    (   DrainResult == drained -> Result = ok ; Result = DrainResult ).

execute_single_hook(Service, health_check, Result) :-
    run_health_check(Service, [retries(3)], HealthResult),
    (   HealthResult == healthy -> Result = ok ; Result = HealthResult ).

execute_single_hook(_Service, save_state, ok) :-
    % Placeholder - would save service state
    true.

execute_single_hook(_Service, warm_cache, ok) :-
    % Placeholder - would warm up caches
    true.

execute_single_hook(Service, custom(Command), Result) :-
    deploy_method_config(Service, Method, MethodOptions),
    service_config(Service, ServiceOptions),

    option_or_default(host, MethodOptions, ServiceOptions, 'localhost', Host),
    option_or_default(user, MethodOptions, ServiceOptions, 'deploy', User),

    (   Method == ssh
    ->  format(atom(Cmd), 'ssh ~w@~w "~w"', [User, Host, Command])
    ;   Cmd = Command
    ),
    (   shell(Cmd, 0)
    ->  Result = ok
    ;   Result = error(command_failed)
    ).

%% ============================================
%% Phase 6b: Health Check Integration
%% ============================================

%% run_health_check(+Service, +Options, -Result)
%  Run health check for a service.
%
%  Options:
%    - retries(N)     : Number of retries (default: 3)
%    - delay(Seconds) : Delay between retries (default: 2)
%    - timeout(Secs)  : Request timeout (default: 5)
%    - endpoint(Path) : Health endpoint (default: '/health')
%
run_health_check(Service, Options, Result) :-
    option_or_default(retries, Options, 3, Retries),
    option_or_default(delay, Options, 2, Delay),
    option_or_default(timeout, Options, 5, Timeout),
    option_or_default(endpoint, Options, '/health', Endpoint),

    service_config(Service, ServiceOptions),
    deploy_method_config(Service, _Method, MethodOptions),

    option_or_default(host, MethodOptions, ServiceOptions, 'localhost', Host),
    option_or_default(port, ServiceOptions, 8080, Port),

    % Determine protocol
    (   is_localhost(Host)
    ->  Protocol = 'http'
    ;   Protocol = 'https'
    ),

    format(atom(URL), '~w://~w:~w~w', [Protocol, Host, Port, Endpoint]),

    run_health_check_loop(URL, Timeout, Retries, Delay, Result).

%% run_health_check_loop(+URL, +Timeout, +Retries, +Delay, -Result)
%  Health check retry loop.
%
run_health_check_loop(URL, Timeout, Retries, Delay, Result) :-
    Retries > 0,
    format(atom(Cmd), 'curl -sf --max-time ~w "~w"', [Timeout, URL]),
    (   shell(Cmd, 0)
    ->  Result = healthy
    ;   NewRetries is Retries - 1,
        (   NewRetries > 0
        ->  sleep(Delay),
            run_health_check_loop(URL, Timeout, NewRetries, Delay, Result)
        ;   Result = unhealthy
        )
    ).

%% wait_for_healthy(+Service, +Options, -Result)
%  Wait for service to become healthy.
%
%  Options:
%    - timeout(Seconds) : Maximum wait time (default: 60)
%    - interval(Secs)   : Check interval (default: 2)
%
wait_for_healthy(Service, Options, Result) :-
    option_or_default(timeout, Options, 60, Timeout),
    option_or_default(interval, Options, 2, Interval),

    MaxRetries is Timeout // Interval,

    run_health_check(Service, [retries(MaxRetries), delay(Interval)], Result).

%% ============================================
%% Phase 6b: Deploy with Full Lifecycle
%% ============================================

%% deploy_with_hooks(+Service, -Result)
%  Deploy service executing all lifecycle hooks.
%
deploy_with_hooks(Service, Result) :-
    % Validate security
    validate_security(Service, SecurityErrors),
    (   SecurityErrors \== []
    ->  Result = error(security_validation_failed(SecurityErrors))
    ;   % Execute pre-deploy hooks (if any)
        execute_hooks(Service, pre_deploy, PreDeployResult),
        (   PreDeployResult \== ok
        ->  Result = error(pre_deploy_failed(PreDeployResult))
        ;   % Check for changes
            check_for_changes(Service, Changes),
            (   Changes == no_changes
            ->  Result = unchanged
            ;   % Backup current deployment for rollback
                backup_current_deployment(Service),

                % Generate and execute deployment
                generate_deploy_script(Service, [], Script),
                execute_deployment(Script, ExecResult),
                (   ExecResult == success
                ->  % Store new hash
                    compute_source_hash(Service, NewHash),
                    store_deployed_hash(Service, NewHash),

                    % Execute post-deploy hooks
                    execute_hooks(Service, post_deploy, PostDeployResult),
                    (   PostDeployResult == ok
                    ->  Result = deployed
                    ;   Result = deployed_with_warnings(PostDeployResult)
                    )
                ;   Result = error(deployment_failed(ExecResult))
                )
            )
        )
    ).

%% backup_current_deployment(+Service)
%  Create backup of current deployment for rollback.
%
backup_current_deployment(Service) :-
    deploy_method_config(Service, Method, MethodOptions),
    service_config(Service, ServiceOptions),

    option_or_default(host, MethodOptions, ServiceOptions, 'localhost', Host),
    option_or_default(user, MethodOptions, ServiceOptions, 'deploy', User),
    option_or_default(remote_dir, MethodOptions, ServiceOptions, '/opt/unifyweaver/services', BaseRemoteDir),

    format(atom(RemoteDir), '~w/~w', [BaseRemoteDir, Service]),

    (   Method == ssh
    ->  format(atom(Cmd), 'ssh ~w@~w "rm -rf ~w.backup; cp -r ~w ~w.backup 2>/dev/null || true"',
               [User, Host, RemoteDir, RemoteDir, RemoteDir]),
        shell(Cmd, _)
    ;   format(atom(Cmd), 'rm -rf .backup; cp -r . .backup 2>/dev/null || true', []),
        shell(Cmd, _)
    ).

%% ============================================
%% Phase 6c: Retry Policies
%% ============================================

%% declare_retry_policy(+Service, +Policy)
%  Declare retry policy for a service.
%
%  Policy options:
%    - max_retries(N)                     : Maximum retry attempts (default: 3)
%    - initial_delay(Ms)                  : Initial delay in milliseconds (default: 1000)
%    - max_delay(Ms)                      : Maximum delay (default: 30000)
%    - backoff(exponential|linear|fixed)  : Backoff strategy (default: exponential)
%    - multiplier(N)                      : Backoff multiplier for exponential (default: 2)
%    - retry_on([Errors])                 : List of error types to retry on
%    - fail_on([Errors])                  : List of error types to fail fast on
%
declare_retry_policy(Service, Policy) :-
    atom(Service),
    is_list(Policy),
    retractall(retry_policy_db(Service, _)),
    assertz(retry_policy_db(Service, Policy)).

%% retry_policy(?Service, ?Policy)
%  Query retry policy for a service.
%
retry_policy(Service, Policy) :-
    retry_policy_db(Service, Policy).

%% call_with_retry(+Service, +Operation, +Args, -Result)
%  Execute an operation with retry policy.
%
%  Operation: Callable term representing the operation to execute
%  Args: Arguments to pass to the operation
%  Result: ok(Value) | error(Reason)
%
call_with_retry(Service, Operation, Args, Result) :-
    (   retry_policy(Service, Policy)
    ->  true
    ;   Policy = [max_retries(3), initial_delay(1000), backoff(exponential)]
    ),

    option_or_default(max_retries, Policy, 3, MaxRetries),
    option_or_default(initial_delay, Policy, 1000, InitialDelay),
    option_or_default(max_delay, Policy, 30000, MaxDelay),
    option_or_default(backoff, Policy, exponential, Backoff),
    option_or_default(multiplier, Policy, 2, Multiplier),
    option_or_default(retry_on, Policy, [], RetryOnErrors),
    option_or_default(fail_on, Policy, [], FailOnErrors),

    retry_loop(Operation, Args, MaxRetries, InitialDelay, MaxDelay,
               Backoff, Multiplier, RetryOnErrors, FailOnErrors, 0, Result).

%% retry_loop/11
%  Internal retry loop.
%
retry_loop(Operation, Args, MaxRetries, CurrentDelay, MaxDelay,
           Backoff, Multiplier, RetryOnErrors, FailOnErrors, Attempt, Result) :-
    Attempt < MaxRetries,
    % Try the operation
    catch(
        (   apply_operation(Operation, Args, OpResult)
        ->  Result = ok(OpResult)
        ;   Result = error(operation_failed)
        ),
        Error,
        handle_retry_error(Error, Operation, Args, MaxRetries, CurrentDelay, MaxDelay,
                          Backoff, Multiplier, RetryOnErrors, FailOnErrors, Attempt, Result)
    ).

retry_loop(_Operation, _Args, MaxRetries, _CurrentDelay, _MaxDelay,
           _Backoff, _Multiplier, _RetryOnErrors, _FailOnErrors, MaxRetries, Result) :-
    Result = error(max_retries_exceeded).

%% handle_retry_error/12
%  Handle an error and decide whether to retry.
%
handle_retry_error(Error, Operation, Args, MaxRetries, CurrentDelay, MaxDelay,
                   Backoff, Multiplier, RetryOnErrors, FailOnErrors, Attempt, Result) :-
    extract_error_type(Error, ErrorType),
    (   % Check if we should fail fast
        (   FailOnErrors \== [],
            member(ErrorType, FailOnErrors)
        )
    ->  Result = error(Error)
    ;   % Check if this error is retryable
        (   RetryOnErrors == []  % Empty means retry on all errors
        ;   member(ErrorType, RetryOnErrors)
        )
    ->  % Calculate next delay
        NextAttempt is Attempt + 1,
        calculate_next_delay(CurrentDelay, MaxDelay, Backoff, Multiplier, NextDelay),

        % Wait before retry
        DelaySeconds is CurrentDelay / 1000,
        sleep(DelaySeconds),

        % Retry
        retry_loop(Operation, Args, MaxRetries, NextDelay, MaxDelay,
                   Backoff, Multiplier, RetryOnErrors, FailOnErrors, NextAttempt, Result)
    ;   % Error not in retry list - fail
        Result = error(Error)
    ).

%% calculate_next_delay(+CurrentDelay, +MaxDelay, +Backoff, +Multiplier, -NextDelay)
%  Calculate the next delay based on backoff strategy.
%
calculate_next_delay(CurrentDelay, MaxDelay, exponential, Multiplier, NextDelay) :-
    NextDelayRaw is CurrentDelay * Multiplier,
    NextDelay is min(NextDelayRaw, MaxDelay).

calculate_next_delay(CurrentDelay, MaxDelay, linear, Multiplier, NextDelay) :-
    NextDelayRaw is CurrentDelay + (Multiplier * 1000),
    NextDelay is min(NextDelayRaw, MaxDelay).

calculate_next_delay(CurrentDelay, _MaxDelay, fixed, _Multiplier, CurrentDelay).

%% extract_error_type(+Error, -Type)
%  Extract the error type for matching.
%
extract_error_type(error(Type, _), Type) :- !.
extract_error_type(error(Type), Type) :- !.
extract_error_type(timeout, timeout) :- !.
extract_error_type(connection_refused, connection_refused) :- !.
extract_error_type(Error, Error).

%% apply_operation(+Operation, +Args, -Result)
%  Apply an operation with arguments.
%
apply_operation(Operation, Args, Result) :-
    (   Args == []
    ->  call(Operation, Result)
    ;   Args = [Arg1]
    ->  call(Operation, Arg1, Result)
    ;   Args = [Arg1, Arg2]
    ->  call(Operation, Arg1, Arg2, Result)
    ;   Args = [Arg1, Arg2, Arg3]
    ->  call(Operation, Arg1, Arg2, Arg3, Result)
    ;   % Fallback: use apply
        append([Operation|Args], [Result], FullCall),
        Call =.. FullCall,
        call(Call)
    ).

%% ============================================
%% Phase 6c: Fallback Mechanisms
%% ============================================

%% declare_fallback(+Service, +Fallback)
%  Declare fallback configuration for a service.
%
%  Fallback options:
%    - backup_service(ServiceName)  : Use another service as fallback
%    - cache(Options)               : Return cached value on failure
%    - default_value(Value)         : Return default value on failure
%    - custom(Predicate)            : Call custom predicate for fallback
%
declare_fallback(Service, Fallback) :-
    atom(Service),
    retractall(fallback_config_db(Service, _)),
    assertz(fallback_config_db(Service, Fallback)).

%% fallback_config(?Service, ?Fallback)
%  Query fallback configuration.
%
fallback_config(Service, Fallback) :-
    fallback_config_db(Service, Fallback).

%% call_with_fallback(+Service, +Operation, +Args, -Result)
%  Execute an operation with fallback on failure.
%
call_with_fallback(Service, Operation, Args, Result) :-
    catch(
        (   apply_operation(Operation, Args, OpResult)
        ->  Result = ok(OpResult)
        ;   execute_fallback(Service, Operation, Args, Result)
        ),
        _Error,
        execute_fallback(Service, Operation, Args, Result)
    ).

%% execute_fallback(+Service, +Operation, +Args, -Result)
%  Execute the configured fallback.
%
execute_fallback(Service, Operation, Args, Result) :-
    (   fallback_config(Service, Fallback)
    ->  apply_fallback(Fallback, Operation, Args, Result)
    ;   Result = error(no_fallback_configured)
    ).

%% apply_fallback(+Fallback, +Operation, +Args, -Result)
%  Apply a specific fallback strategy.
%
apply_fallback(backup_service(BackupService), Operation, Args, Result) :-
    % Try the backup service
    catch(
        (   apply_operation_for_service(BackupService, Operation, Args, OpResult)
        ->  Result = ok(OpResult)
        ;   Result = error(backup_service_failed)
        ),
        Error,
        Result = error(backup_service_error(Error))
    ).

apply_fallback(cache(CacheOptions), _Operation, _Args, Result) :-
    option_or_default(key, CacheOptions, default, CacheKey),
    option_or_default(ttl, CacheOptions, 3600, _TTL),
    (   get_cached_value(CacheKey, CachedValue)
    ->  Result = ok(CachedValue)
    ;   Result = error(cache_miss)
    ).

apply_fallback(default_value(Value), _Operation, _Args, ok(Value)).

apply_fallback(custom(Predicate), Operation, Args, Result) :-
    catch(
        (   call(Predicate, Operation, Args, FallbackResult)
        ->  Result = ok(FallbackResult)
        ;   Result = error(custom_fallback_failed)
        ),
        Error,
        Result = error(custom_fallback_error(Error))
    ).

%% apply_operation_for_service(+Service, +Operation, +Args, -Result)
%  Apply an operation in the context of a specific service.
%
apply_operation_for_service(_Service, Operation, Args, Result) :-
    apply_operation(Operation, Args, Result).

%% get_cached_value(+Key, -Value)
%  Get a cached value (placeholder - would integrate with actual cache).
%
get_cached_value(_Key, _Value) :-
    fail.  % No cache implementation yet

%% ============================================
%% Phase 6c: Circuit Breaker
%% ============================================

%% declare_circuit_breaker(+Service, +Config)
%  Declare circuit breaker configuration for a service.
%
%  Config options:
%    - failure_threshold(N)    : Number of failures before opening (default: 5)
%    - success_threshold(N)    : Number of successes to close (default: 3)
%    - half_open_timeout(Ms)   : Time before trying half-open (default: 30000)
%    - reset_timeout(Ms)       : Time before auto-reset (default: 60000)
%
declare_circuit_breaker(Service, Config) :-
    atom(Service),
    is_list(Config),
    retractall(circuit_breaker_db(Service, _)),
    assertz(circuit_breaker_db(Service, Config)),
    % Initialize state to closed
    reset_circuit_breaker(Service).

%% circuit_breaker_config(?Service, ?Config)
%  Query circuit breaker configuration.
%
circuit_breaker_config(Service, Config) :-
    circuit_breaker_db(Service, Config).

%% circuit_state(?Service, ?State)
%  Query circuit breaker state.
%  State: closed | open | half_open
%
circuit_state(Service, State) :-
    (   circuit_state_db(Service, State, _Data)
    ->  true
    ;   State = closed  % Default
    ).

%% reset_circuit_breaker(+Service)
%  Reset circuit breaker to closed state.
%
reset_circuit_breaker(Service) :-
    retractall(circuit_state_db(Service, _, _)),
    assertz(circuit_state_db(Service, closed, state_data(0, 0, 0))).

%% call_with_circuit_breaker(+Service, +Operation, +Args, -Result)
%  Execute an operation with circuit breaker protection.
%
call_with_circuit_breaker(Service, Operation, Args, Result) :-
    circuit_state(Service, CurrentState),
    (   CurrentState == open
    ->  % Check if we should try half-open
        check_half_open_transition(Service, ShouldTry),
        (   ShouldTry == yes
        ->  execute_with_circuit(Service, Operation, Args, Result)
        ;   Result = error(circuit_open)
        )
    ;   execute_with_circuit(Service, Operation, Args, Result)
    ).

%% check_half_open_transition(+Service, -ShouldTry)
%  Check if circuit should transition to half-open.
%
check_half_open_transition(Service, ShouldTry) :-
    (   circuit_breaker_config(Service, Config)
    ->  true
    ;   Config = []
    ),
    option_or_default(half_open_timeout, Config, 30000, HalfOpenTimeout),

    circuit_state_db(Service, open, state_data(_, _, LastFailureTime)),
    get_time(Now),
    NowMs is Now * 1000,
    TimeSinceFailure is NowMs - LastFailureTime,

    (   TimeSinceFailure >= HalfOpenTimeout
    ->  % Transition to half-open
        retractall(circuit_state_db(Service, _, _)),
        assertz(circuit_state_db(Service, half_open, state_data(0, 0, LastFailureTime))),
        ShouldTry = yes
    ;   ShouldTry = no
    ).

%% execute_with_circuit(+Service, +Operation, +Args, -Result)
%  Execute operation and update circuit state.
%
execute_with_circuit(Service, Operation, Args, Result) :-
    catch(
        (   apply_operation(Operation, Args, OpResult)
        ->  record_circuit_success(Service),
            Result = ok(OpResult)
        ;   record_circuit_failure(Service),
            Result = error(operation_failed)
        ),
        Error,
        (   record_circuit_failure(Service),
            Result = error(Error)
        )
    ).

%% record_circuit_success(+Service)
%  Record a successful call for circuit breaker.
%
record_circuit_success(Service) :-
    (   circuit_breaker_config(Service, Config)
    ->  true
    ;   Config = []
    ),
    option_or_default(success_threshold, Config, 3, SuccessThreshold),

    circuit_state_db(Service, State, state_data(Failures, Successes, LastTime)),
    NewSuccesses is Successes + 1,

    (   State == half_open, NewSuccesses >= SuccessThreshold
    ->  % Close the circuit
        retractall(circuit_state_db(Service, _, _)),
        assertz(circuit_state_db(Service, closed, state_data(0, 0, LastTime)))
    ;   retractall(circuit_state_db(Service, _, _)),
        assertz(circuit_state_db(Service, State, state_data(Failures, NewSuccesses, LastTime)))
    ).

%% record_circuit_failure(+Service)
%  Record a failed call for circuit breaker.
%
record_circuit_failure(Service) :-
    (   circuit_breaker_config(Service, Config)
    ->  true
    ;   Config = []
    ),
    option_or_default(failure_threshold, Config, 5, FailureThreshold),

    circuit_state_db(Service, State, state_data(Failures, Successes, _LastTime)),
    NewFailures is Failures + 1,
    get_time(Now),
    NowMs is Now * 1000,

    (   State == half_open
    ->  % Back to open on any failure in half-open
        retractall(circuit_state_db(Service, _, _)),
        assertz(circuit_state_db(Service, open, state_data(NewFailures, 0, NowMs)))
    ;   NewFailures >= FailureThreshold
    ->  % Open the circuit
        retractall(circuit_state_db(Service, _, _)),
        assertz(circuit_state_db(Service, open, state_data(NewFailures, 0, NowMs)))
    ;   retractall(circuit_state_db(Service, _, _)),
        assertz(circuit_state_db(Service, State, state_data(NewFailures, Successes, NowMs)))
    ).

%% ============================================
%% Phase 6c: Timeout Configuration
%% ============================================

%% declare_timeouts(+Service, +Timeouts)
%  Declare timeout configuration for a service.
%
%  Timeout options:
%    - connect_timeout(Ms)  : Connection timeout (default: 5000)
%    - read_timeout(Ms)     : Read/response timeout (default: 30000)
%    - total_timeout(Ms)    : Total operation timeout (default: 60000)
%    - idle_timeout(Ms)     : Idle connection timeout (default: 120000)
%
declare_timeouts(Service, Timeouts) :-
    atom(Service),
    is_list(Timeouts),
    retractall(timeout_config_db(Service, _)),
    assertz(timeout_config_db(Service, Timeouts)).

%% timeout_config(?Service, ?Timeouts)
%  Query timeout configuration.
%
timeout_config(Service, Timeouts) :-
    timeout_config_db(Service, Timeouts).

%% call_with_timeout(+Service, +Operation, +Args, -Result)
%  Execute an operation with configured timeout.
%
call_with_timeout(Service, Operation, Args, Result) :-
    (   timeout_config(Service, Timeouts)
    ->  true
    ;   Timeouts = [total_timeout(60000)]
    ),
    option_or_default(total_timeout, Timeouts, 60000, TotalTimeoutMs),

    TotalTimeoutSecs is TotalTimeoutMs / 1000,

    catch(
        call_with_time_limit(TotalTimeoutSecs,
            (   apply_operation(Operation, Args, OpResult)
            ->  Result = ok(OpResult)
            ;   Result = error(operation_failed)
            )
        ),
        time_limit_exceeded,
        Result = error(timeout)
    ).

%% call_with_time_limit(+Seconds, +Goal)
%  Execute goal with time limit.
%
call_with_time_limit(Seconds, Goal) :-
    catch(
        setup_call_cleanup(
            alarm(Seconds, throw(time_limit_exceeded), AlarmId),
            Goal,
            remove_alarm(AlarmId)
        ),
        time_limit_exceeded,
        throw(time_limit_exceeded)
    ).

%% ============================================
%% Phase 6c: Combined Error Handling
%% ============================================

%% protected_call(+Service, +Operation, +Args, -Result)
%  Execute an operation with full error handling:
%  1. Check circuit breaker
%  2. Apply timeouts
%  3. Retry on failure
%  4. Fall back if all retries fail
%
protected_call(Service, Operation, Args, Result) :-
    % Check circuit breaker first
    circuit_state(Service, CircuitState),
    (   CircuitState == open
    ->  check_half_open_transition(Service, ShouldTry),
        (   ShouldTry == yes
        ->  protected_call_internal(Service, Operation, Args, Result)
        ;   % Circuit is open - try fallback directly
            execute_fallback(Service, Operation, Args, Result)
        )
    ;   protected_call_internal(Service, Operation, Args, Result)
    ).

%% protected_call_internal(+Service, +Operation, +Args, -Result)
%  Internal protected call with retry and fallback.
%
protected_call_internal(Service, Operation, Args, Result) :-
    % Try operation with timeout directly (retry wrapping is complex, simplify)
    (   timeout_config(Service, _Timeouts)
    ->  % Use timeout-wrapped call
        call_with_timeout(Service, Operation, Args, TimeoutResult),
        (   TimeoutResult = ok(Value)
        ->  record_circuit_success(Service),
            Result = ok(Value)
        ;   record_circuit_failure(Service),
            execute_fallback(Service, Operation, Args, FallbackResult),
            (   FallbackResult = ok(_)
            ->  Result = FallbackResult
            ;   Result = TimeoutResult
            )
        )
    ;   % No timeout configured - direct call with retry
        call_with_retry(Service, Operation, Args, RetryResult),
        (   RetryResult = ok(Value)
        ->  record_circuit_success(Service),
            Result = ok(Value)
        ;   record_circuit_failure(Service),
            execute_fallback(Service, Operation, Args, FallbackResult),
            (   FallbackResult = ok(_)
            ->  Result = FallbackResult
            ;   Result = RetryResult
            )
        )
    ).

%% call_with_timeout_internal(+Service, +Operation, +Args, -Result)
%  Internal timeout wrapper for use in retry loop.
%
call_with_timeout_internal(Service, Operation, Args, Result) :-
    call_with_timeout(Service, Operation, Args, TimeoutResult),
    (   TimeoutResult = ok(Value)
    ->  Result = Value
    ;   TimeoutResult = error(Error)
    ->  throw(Error)
    ).

%% ============================================
%% Phase 6d: Health Check Monitoring
%% ============================================

%% declare_health_check(+Service, +Config)
%  Declare health check configuration for a service.
%
%  Config options:
%    - endpoint(Path)           : Health endpoint (default: '/health')
%    - interval(Seconds)        : Check interval (default: 30)
%    - timeout(Seconds)         : Request timeout (default: 5)
%    - unhealthy_threshold(N)   : Failures before unhealthy (default: 3)
%    - healthy_threshold(N)     : Successes before healthy (default: 2)
%
declare_health_check(Service, Config) :-
    atom(Service),
    is_list(Config),
    retractall(health_check_config_db(Service, _)),
    assertz(health_check_config_db(Service, Config)).

%% health_check_config(?Service, ?Config)
%  Query health check configuration.
%
health_check_config(Service, Config) :-
    health_check_config_db(Service, Config).

%% health_status(?Service, ?Status)
%  Query current health status of a service.
%  Status: healthy | unhealthy | unknown
%
health_status(Service, Status) :-
    (   health_status_db(Service, Status, _Timestamp)
    ->  true
    ;   Status = unknown
    ).

%% update_health_status(+Service, +Status)
%  Update health status for a service.
%
update_health_status(Service, Status) :-
    get_time(Now),
    retractall(health_status_db(Service, _, _)),
    assertz(health_status_db(Service, Status, Now)),
    % Log the status change
    (   logging_config(Service, _)
    ->  log_event(Service, info, 'Health status changed', [status-Status])
    ;   true
    ),
    % Check if we need to trigger alerts
    check_health_alerts(Service, Status).

%% check_health_alerts(+Service, +Status)
%  Check and trigger health-related alerts.
%
check_health_alerts(Service, unhealthy) :-
    (   alert_config(Service, service_unhealthy, _)
    ->  trigger_alert(Service, service_unhealthy, [status-unhealthy])
    ;   true
    ).
check_health_alerts(Service, healthy) :-
    (   alert_state_db(Service, service_unhealthy, triggered, _)
    ->  resolve_alert(Service, service_unhealthy)
    ;   true
    ).
check_health_alerts(_, _).

%% start_health_monitor(+Service, -Result)
%  Start background health monitoring for a service.
%  Note: This is a simplified version - in production would use threads.
%
start_health_monitor(Service, Result) :-
    (   health_check_config(Service, _Config)
    ->  % For now, just mark as started and do initial check
        run_health_check(Service, [], HealthResult),
        (   HealthResult == healthy
        ->  update_health_status(Service, healthy)
        ;   update_health_status(Service, unhealthy)
        ),
        Result = started
    ;   Result = error(no_health_check_configured)
    ).

%% stop_health_monitor(+Service)
%  Stop health monitoring for a service.
%
stop_health_monitor(Service) :-
    retractall(health_monitor_pid_db(Service, _)).

%% ============================================
%% Phase 6d: Metrics Collection
%% ============================================

%% declare_metrics(+Service, +Config)
%  Declare metrics configuration for a service.
%
%  Config options:
%    - collect([Metrics])       : List of metrics to collect
%    - labels([Labels])         : Labels to attach to metrics
%    - export(Format)           : prometheus | statsd | json
%    - port(Port)               : Metrics server port
%    - retention(Seconds)       : How long to keep metrics (default: 3600)
%
declare_metrics(Service, Config) :-
    atom(Service),
    is_list(Config),
    retractall(metrics_config_db(Service, _)),
    assertz(metrics_config_db(Service, Config)).

%% metrics_config(?Service, ?Config)
%  Query metrics configuration.
%
metrics_config(Service, Config) :-
    metrics_config_db(Service, Config).

%% record_metric(+Service, +Metric, +Value)
%  Record a metric value for a service.
%
%  Metric types:
%    - counter(Name)            : Incrementing counter
%    - gauge(Name)              : Point-in-time value
%    - histogram(Name, Bucket)  : Distribution of values
%
record_metric(Service, Metric, Value) :-
    get_time(Now),
    assertz(metric_data_db(Service, Metric, Value, Now)),
    % Clean old metrics based on retention
    clean_old_metrics(Service).

%% clean_old_metrics(+Service)
%  Remove metrics older than retention period.
%
clean_old_metrics(Service) :-
    (   metrics_config(Service, Config)
    ->  option_or_default(retention, Config, 3600, Retention)
    ;   Retention = 3600
    ),
    get_time(Now),
    Cutoff is Now - Retention,
    forall(
        (   metric_data_db(Service, Metric, Value, Timestamp),
            Timestamp < Cutoff
        ),
        retract(metric_data_db(Service, Metric, Value, Timestamp))
    ).

%% get_metrics(+Service, -Metrics)
%  Get all current metrics for a service.
%
get_metrics(Service, Metrics) :-
    findall(
        metric(Metric, Value, Timestamp),
        metric_data_db(Service, Metric, Value, Timestamp),
        Metrics
    ).

%% generate_prometheus_metrics(+Service, -Output)
%  Generate Prometheus-compatible metrics output.
%
generate_prometheus_metrics(Service, Output) :-
    get_metrics(Service, Metrics),
    (   metrics_config(Service, Config)
    ->  option_or_default(labels, Config, [], Labels)
    ;   Labels = []
    ),
    format_prometheus_metrics(Service, Metrics, Labels, Output).

%% format_prometheus_metrics(+Service, +Metrics, +Labels, -Output)
%  Format metrics in Prometheus text format.
%
format_prometheus_metrics(Service, Metrics, Labels, Output) :-
    aggregate_metrics(Metrics, Aggregated),
    format_labels(Labels, LabelStr),
    maplist(format_single_prometheus_metric(Service, LabelStr), Aggregated, Lines),
    atomic_list_concat(Lines, '\n', Output).

%% aggregate_metrics(+Metrics, -Aggregated)
%  Aggregate metrics by name (latest value for gauges, sum for counters).
%
aggregate_metrics(Metrics, Aggregated) :-
    findall(Name, member(metric(Name, _, _), Metrics), Names),
    sort(Names, UniqueNames),
    maplist(aggregate_single_metric(Metrics), UniqueNames, Aggregated).

aggregate_single_metric(Metrics, Name, aggregated(Name, Value, Count)) :-
    findall(V, member(metric(Name, V, _), Metrics), Values),
    length(Values, Count),
    (   Count > 0
    ->  last(Values, Value)  % Use latest value
    ;   Value = 0
    ).

%% format_labels(+Labels, -LabelStr)
%  Format labels for Prometheus.
%
format_labels([], '').
format_labels(Labels, LabelStr) :-
    Labels \== [],
    maplist(format_single_label, Labels, LabelParts),
    atomic_list_concat(LabelParts, ',', Inner),
    format(atom(LabelStr), '{~w}', [Inner]).

format_single_label(Label, Part) :-
    (   Label = Name-Value
    ->  format(atom(Part), '~w="~w"', [Name, Value])
    ;   format(atom(Part), '~w', [Label])
    ).

%% format_single_prometheus_metric(+Service, +LabelStr, +Aggregated, -Line)
%  Format a single metric line.
%
format_single_prometheus_metric(Service, LabelStr, aggregated(Name, Value, _Count), Line) :-
    (   LabelStr == ''
    ->  format(atom(Line), '~w_~w ~w', [Service, Name, Value])
    ;   format(atom(Line), '~w_~w~w ~w', [Service, Name, LabelStr, Value])
    ).

%% ============================================
%% Phase 6d: Structured Logging
%% ============================================

%% declare_logging(+Service, +Config)
%  Declare logging configuration for a service.
%
%  Config options:
%    - level(Level)             : debug | info | warn | error (default: info)
%    - format(Format)           : json | text (default: json)
%    - output(Output)           : stdout | file(Path) (default: stdout)
%    - include([Fields])        : Fields to include in each entry
%    - max_entries(N)           : Max log entries to keep (default: 1000)
%
declare_logging(Service, Config) :-
    atom(Service),
    is_list(Config),
    retractall(logging_config_db(Service, _)),
    assertz(logging_config_db(Service, Config)).

%% logging_config(?Service, ?Config)
%  Query logging configuration.
%
logging_config(Service, Config) :-
    logging_config_db(Service, Config).

%% log_event(+Service, +Level, +Message, +Data)
%  Log an event for a service.
%
%  Level: debug | info | warn | error
%  Data: List of Key-Value pairs
%
log_event(Service, Level, Message, Data) :-
    (   logging_config(Service, Config)
    ->  option_or_default(level, Config, info, MinLevel),
        (   level_priority(Level, LevelPri),
            level_priority(MinLevel, MinPri),
            LevelPri >= MinPri
        ->  get_time(Now),
            assertz(log_entry_db(Service, Level, Message, Data, Now)),
            output_log_entry(Service, Config, Level, Message, Data, Now),
            clean_old_logs(Service, Config)
        ;   true  % Level below minimum, skip
        )
    ;   % No logging configured, still store internally
        get_time(Now),
        assertz(log_entry_db(Service, Level, Message, Data, Now))
    ).

%% level_priority(+Level, -Priority)
%  Get priority for log level (higher = more severe).
%
level_priority(debug, 0).
level_priority(info, 1).
level_priority(warn, 2).
level_priority(warning, 2).
level_priority(error, 3).

%% output_log_entry(+Service, +Config, +Level, +Message, +Data, +Timestamp)
%  Output log entry based on configuration.
%
output_log_entry(Service, Config, Level, Message, Data, Timestamp) :-
    option_or_default(format, Config, json, Format),
    option_or_default(output, Config, stdout, Output),
    format_log_entry(Format, Service, Level, Message, Data, Timestamp, Formatted),
    write_log_output(Output, Formatted).

%% format_log_entry(+Format, +Service, +Level, +Message, +Data, +Timestamp, -Formatted)
%  Format log entry.
%
format_log_entry(json, Service, Level, Message, Data, Timestamp, Formatted) :-
    format_timestamp_iso(Timestamp, ISOTime),
    format_json_data(Data, JsonData),
    format(atom(Formatted),
           '{"timestamp":"~w","service":"~w","level":"~w","message":"~w"~w}',
           [ISOTime, Service, Level, Message, JsonData]).

format_log_entry(text, Service, Level, Message, Data, Timestamp, Formatted) :-
    format_timestamp_iso(Timestamp, ISOTime),
    format_text_data(Data, TextData),
    format(atom(Formatted),
           '[~w] ~w [~w] ~w~w',
           [ISOTime, Level, Service, Message, TextData]).

%% format_timestamp_iso(+Timestamp, -ISO)
%  Format timestamp as ISO 8601.
%
format_timestamp_iso(Timestamp, ISO) :-
    stamp_date_time(Timestamp, DateTime, 'UTC'),
    format_time(atom(ISO), '%FT%T%:z', DateTime).

%% format_json_data(+Data, -JsonStr)
%  Format data as JSON fields.
%
format_json_data([], '').
format_json_data(Data, JsonStr) :-
    Data \== [],
    maplist(format_json_field, Data, Fields),
    atomic_list_concat(Fields, '', FieldStr),
    format(atom(JsonStr), ',~w', [FieldStr]).

format_json_field(Key-Value, Field) :-
    format(atom(Field), '"~w":"~w"', [Key, Value]).

%% format_text_data(+Data, -TextStr)
%  Format data as text fields.
%
format_text_data([], '').
format_text_data(Data, TextStr) :-
    Data \== [],
    maplist(format_text_field, Data, Fields),
    atomic_list_concat(Fields, ' ', FieldStr),
    format(atom(TextStr), ' [~w]', [FieldStr]).

format_text_field(Key-Value, Field) :-
    format(atom(Field), '~w=~w', [Key, Value]).

%% write_log_output(+Output, +Formatted)
%  Write formatted log to output.
%
write_log_output(stdout, Formatted) :-
    format('~w~n', [Formatted]).
write_log_output(file(Path), Formatted) :-
    open(Path, append, Stream),
    format(Stream, '~w~n', [Formatted]),
    close(Stream).

%% clean_old_logs(+Service, +Config)
%  Remove old log entries beyond max_entries.
%
clean_old_logs(Service, Config) :-
    option_or_default(max_entries, Config, 1000, MaxEntries),
    findall(T, log_entry_db(Service, _, _, _, T), Timestamps),
    length(Timestamps, Count),
    (   Count > MaxEntries
    ->  ToRemove is Count - MaxEntries,
        sort(Timestamps, Sorted),
        length(OldTimestamps, ToRemove),
        append(OldTimestamps, _, Sorted),
        forall(member(T, OldTimestamps),
               retract(log_entry_db(Service, _, _, _, T)))
    ;   true
    ).

%% get_log_entries(+Service, +Options, -Entries)
%  Get log entries for a service.
%
%  Options:
%    - level(Level)     : Filter by minimum level
%    - limit(N)         : Maximum entries to return
%    - since(Timestamp) : Entries since timestamp
%
get_log_entries(Service, Options, Entries) :-
    findall(
        entry(Level, Message, Data, Timestamp),
        (   log_entry_db(Service, Level, Message, Data, Timestamp),
            filter_log_entry(Options, Level, Timestamp)
        ),
        AllEntries
    ),
    option_or_default(limit, Options, 100, Limit),
    (   length(AllEntries, Len), Len > Limit
    ->  length(Entries, Limit),
        append(_, Entries, AllEntries)  % Take last N
    ;   Entries = AllEntries
    ).

%% filter_log_entry(+Options, +Level, +Timestamp)
%  Check if log entry matches filters.
%
filter_log_entry(Options, Level, Timestamp) :-
    (   member(level(MinLevel), Options)
    ->  level_priority(Level, LevelPri),
        level_priority(MinLevel, MinPri),
        LevelPri >= MinPri
    ;   true
    ),
    (   member(since(Since), Options)
    ->  Timestamp >= Since
    ;   true
    ).

%% ============================================
%% Phase 6d: Alerting
%% ============================================

%% declare_alert(+Service, +AlertName, +Config)
%  Declare an alert for a service.
%
%  Config options:
%    - condition(Condition)     : Alert condition (term or string)
%    - duration(Seconds)        : How long condition must hold (default: 0)
%    - severity(Level)          : critical | warning | info (default: warning)
%    - notify([Channels])       : Notification channels
%    - cooldown(Seconds)        : Minimum time between alerts (default: 300)
%
declare_alert(Service, AlertName, Config) :-
    atom(Service),
    atom(AlertName),
    is_list(Config),
    retractall(alert_config_db(Service, AlertName, _)),
    assertz(alert_config_db(Service, AlertName, Config)),
    % Initialize alert state
    retractall(alert_state_db(Service, AlertName, _, _)),
    assertz(alert_state_db(Service, AlertName, resolved, 0)).

%% alert_config(?Service, ?AlertName, ?Config)
%  Query alert configuration.
%
alert_config(Service, AlertName, Config) :-
    alert_config_db(Service, AlertName, Config).

%% trigger_alert(+Service, +AlertName, +Data)
%  Trigger an alert.
%
trigger_alert(Service, AlertName, Data) :-
    get_time(Now),
    (   alert_config(Service, AlertName, Config)
    ->  option_or_default(severity, Config, warning, Severity),
        option_or_default(cooldown, Config, 300, Cooldown),

        % Check cooldown
        (   alert_state_db(Service, AlertName, triggered, LastTime),
            Now - LastTime < Cooldown
        ->  true  % Still in cooldown, skip
        ;   % Trigger the alert
            retractall(alert_state_db(Service, AlertName, _, _)),
            assertz(alert_state_db(Service, AlertName, triggered, Now)),
            % Record in history
            assertz(alert_history_db(Service, AlertName, triggered, Data, Now)),
            % Log the alert
            log_event(Service, Severity, 'Alert triggered', [alert-AlertName|Data]),
            % Send notifications
            send_alert_notifications(Service, AlertName, Config, Data)
        )
    ;   true  % No such alert configured
    ).

%% resolve_alert(+Service, +AlertName)
%  Resolve a triggered alert.
%
resolve_alert(Service, AlertName) :-
    get_time(Now),
    (   alert_state_db(Service, AlertName, triggered, _)
    ->  retractall(alert_state_db(Service, AlertName, _, _)),
        assertz(alert_state_db(Service, AlertName, resolved, Now)),
        assertz(alert_history_db(Service, AlertName, resolved, [], Now)),
        log_event(Service, info, 'Alert resolved', [alert-AlertName])
    ;   true
    ).

%% send_alert_notifications(+Service, +AlertName, +Config, +Data)
%  Send notifications for an alert.
%
send_alert_notifications(Service, AlertName, Config, Data) :-
    (   member(notify(Channels), Config)
    ->  maplist(send_notification(Service, AlertName, Data), Channels)
    ;   true  % No notification channels configured
    ).

%% send_notification(+Service, +AlertName, +Data, +Channel)
%  Send notification to a specific channel.
%
send_notification(Service, AlertName, Data, slack(Channel)) :-
    format(user_error, '[SLACK ~w] Alert ~w/~w: ~w~n',
           [Channel, Service, AlertName, Data]).
send_notification(Service, AlertName, Data, email(Address)) :-
    format(user_error, '[EMAIL ~w] Alert ~w/~w: ~w~n',
           [Address, Service, AlertName, Data]).
send_notification(Service, AlertName, Data, pagerduty) :-
    format(user_error, '[PAGERDUTY] Alert ~w/~w: ~w~n',
           [Service, AlertName, Data]).
send_notification(Service, AlertName, Data, webhook(URL)) :-
    format(user_error, '[WEBHOOK ~w] Alert ~w/~w: ~w~n',
           [URL, Service, AlertName, Data]).
send_notification(_, _, _, _).  % Unknown channel - ignore

%% check_alerts(+Service, -TriggeredAlerts)
%  Check all alerts for a service and return triggered ones.
%
check_alerts(Service, TriggeredAlerts) :-
    findall(
        alert(AlertName, State, Since),
        (   alert_state_db(Service, AlertName, State, Since),
            State == triggered
        ),
        TriggeredAlerts
    ).

%% alert_history(+Service, +Options, -History)
%  Get alert history for a service.
%
%  Options:
%    - alert(Name)       : Filter by alert name
%    - limit(N)          : Maximum entries
%    - since(Timestamp)  : Entries since timestamp
%
alert_history(Service, Options, History) :-
    findall(
        history(AlertName, State, Data, Timestamp),
        (   alert_history_db(Service, AlertName, State, Data, Timestamp),
            filter_alert_history(Options, AlertName, Timestamp)
        ),
        AllHistory
    ),
    option_or_default(limit, Options, 100, Limit),
    (   length(AllHistory, Len), Len > Limit
    ->  length(History, Limit),
        append(_, History, AllHistory)
    ;   History = AllHistory
    ).

%% filter_alert_history(+Options, +AlertName, +Timestamp)
%  Check if alert history entry matches filters.
%
filter_alert_history(Options, AlertName, Timestamp) :-
    (   member(alert(FilterName), Options)
    ->  AlertName == FilterName
    ;   true
    ),
    (   member(since(Since), Options)
    ->  Timestamp >= Since
    ;   true
    ).
