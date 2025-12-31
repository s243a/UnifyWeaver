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
 *
 * EXPERIMENTAL FEATURES (Phase 7a - Container Deployment):
 * The following features are experimental and have only been tested for
 * code generation correctness, NOT actual deployment functionality:
 * - Docker configuration and Dockerfile generation
 * - Docker Compose generation
 * - Kubernetes manifest generation (Deployment, Service, Ingress, ConfigMap)
 * - Container registry authentication
 * - Helm chart generation
 *
 * EXPERIMENTAL FEATURES (Phase 7b - Secrets Management):
 * - HashiCorp Vault integration (read, agent config)
 * - AWS Secrets Manager integration
 * - Azure Key Vault integration
 * - GCP Secret Manager integration
 * - Service secret bindings and environment injection
 * - Kubernetes Secret and ExternalSecret manifests
 *
 * EXPERIMENTAL FEATURES (Phase 7c - Multi-Region & Cloud Functions):
 * - Multi-region deployment configuration
 * - Geographic failover configuration
 * - Health-based region selection
 * - AWS Lambda function deployment
 * - Google Cloud Functions deployment
 * - Azure Functions deployment
 * - API Gateway integration (AWS, GCP, Azure)
 *
 * These predicates generate shell commands and YAML/config content but
 * do NOT execute them. Integration testing with actual cloud services
 * is required before production use.
 *
 * See TEST_COVERAGE.md for detailed test coverage information.
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
    alert_history/3,                % alert_history(+Service, +Options, -History)

    % Phase 7a: Container Deployment
    % Docker configuration
    declare_docker_config/2,        % declare_docker_config(+Service, +Config)
    docker_config/2,                % docker_config(?Service, ?Config)

    % Dockerfile generation
    generate_dockerfile/3,          % generate_dockerfile(+Service, +Options, -Dockerfile)
    generate_dockerignore/3,        % generate_dockerignore(+Service, +Options, -Content)

    % Docker operations
    build_docker_image/3,           % build_docker_image(+Service, +Options, -Result)
    push_docker_image/3,            % push_docker_image(+Service, +Options, -Result)
    docker_image_tag/2,             % docker_image_tag(+Service, -Tag)

    % Docker Compose
    declare_compose_config/2,       % declare_compose_config(+Project, +Config)
    compose_config/2,               % compose_config(?Project, ?Config)
    generate_docker_compose/3,      % generate_docker_compose(+Project, +Options, -ComposeYaml)

    % Kubernetes deployment
    declare_k8s_config/2,           % declare_k8s_config(+Service, +Config)
    k8s_config/2,                   % k8s_config(?Service, ?Config)
    generate_k8s_deployment/3,      % generate_k8s_deployment(+Service, +Options, -Manifest)
    generate_k8s_service/3,         % generate_k8s_service(+Service, +Options, -Manifest)
    generate_k8s_configmap/3,       % generate_k8s_configmap(+Service, +Options, -Manifest)
    generate_k8s_ingress/3,         % generate_k8s_ingress(+Service, +Options, -Manifest)
    generate_helm_chart/3,          % generate_helm_chart(+Service, +Options, -Chart)

    % Container registry
    declare_registry/2,             % declare_registry(+Name, +Config)
    registry_config/2,              % registry_config(?Name, ?Config)
    login_registry/2,               % login_registry(+Name, -Result)

    % Container orchestration
    deploy_to_k8s/3,                % deploy_to_k8s(+Service, +Options, -Result)
    scale_k8s_deployment/4,         % scale_k8s_deployment(+Service, +Replicas, +Options, -Result)
    rollout_status/3,               % rollout_status(+Service, +Options, -Status)

    % Phase 7b: Secrets Management [EXPERIMENTAL]
    % Secret source declarations
    declare_secret_source/2,        % declare_secret_source(+Name, +Config)
    secret_source_config/2,         % secret_source_config(?Name, ?Config)

    % HashiCorp Vault
    declare_vault_config/2,         % declare_vault_config(+Name, +Config)
    vault_config/2,                 % vault_config(?Name, ?Config)
    generate_vault_read/4,          % generate_vault_read(+Source, +Path, +Options, -Command)
    generate_vault_agent_config/3,  % generate_vault_agent_config(+Service, +Options, -Config)

    % AWS Secrets Manager
    declare_aws_secrets_config/2,   % declare_aws_secrets_config(+Name, +Config)
    aws_secrets_config/2,           % aws_secrets_config(?Name, ?Config)
    generate_aws_secret_read/4,     % generate_aws_secret_read(+Source, +SecretId, +Options, -Command)

    % Azure Key Vault
    declare_azure_keyvault_config/2, % declare_azure_keyvault_config(+Name, +Config)
    azure_keyvault_config/2,        % azure_keyvault_config(?Name, ?Config)
    generate_azure_secret_read/4,   % generate_azure_secret_read(+Source, +SecretName, +Options, -Command)

    % GCP Secret Manager
    declare_gcp_secrets_config/2,   % declare_gcp_secrets_config(+Name, +Config)
    gcp_secrets_config/2,           % gcp_secrets_config(?Name, ?Config)
    generate_gcp_secret_read/4,     % generate_gcp_secret_read(+Source, +SecretId, +Options, -Command)

    % Secret bindings for services
    declare_service_secrets/2,      % declare_service_secrets(+Service, +Secrets)
    service_secrets/2,              % service_secrets(?Service, ?Secrets)

    % Environment variable injection
    generate_secret_env_script/3,   % generate_secret_env_script(+Service, +Options, -Script)
    generate_k8s_secret/3,          % generate_k8s_secret(+Service, +Options, -Manifest)
    generate_k8s_external_secret/3, % generate_k8s_external_secret(+Service, +Options, -Manifest)

    % Secret rotation
    declare_secret_rotation/3,      % declare_secret_rotation(+Service, +Secret, +Config)
    secret_rotation_config/3,       % secret_rotation_config(?Service, ?Secret, ?Config)

    % Unified secret access
    resolve_secret/4,               % resolve_secret(+Source, +Path, +Options, -Result)
    list_secrets/3,                 % list_secrets(+Source, +Options, -Secrets)

    % Phase 7c: Multi-Region Deployment [EXPERIMENTAL]
    % Region configuration
    declare_region/2,               % declare_region(+Name, +Config)
    region_config/2,                % region_config(?Name, ?Config)
    declare_service_regions/2,      % declare_service_regions(+Service, +RegionConfig)
    service_regions/2,              % service_regions(?Service, ?RegionConfig)

    % Geographic failover
    declare_failover_policy/2,      % declare_failover_policy(+Service, +Policy)
    failover_policy/2,              % failover_policy(?Service, ?Policy)
    select_region/3,                % select_region(+Service, +Options, -Region)
    failover_to_region/3,           % failover_to_region(+Service, +Region, -Result)

    % Multi-region deployment
    deploy_to_region/4,             % deploy_to_region(+Service, +Region, +Options, -Result)
    deploy_to_all_regions/3,        % deploy_to_all_regions(+Service, +Options, -Results)
    region_status/3,                % region_status(+Service, +Region, -Status)
    generate_region_config/3,       % generate_region_config(+Service, +Options, -Config)

    % Traffic management
    declare_traffic_policy/2,       % declare_traffic_policy(+Service, +Policy)
    traffic_policy/2,               % traffic_policy(?Service, ?Policy)
    generate_route53_config/3,      % generate_route53_config(+Service, +Options, -Config)
    generate_cloudflare_config/3,   % generate_cloudflare_config(+Service, +Options, -Config)

    % Phase 7c: Cloud Functions [EXPERIMENTAL]
    % AWS Lambda
    declare_lambda_config/2,        % declare_lambda_config(+Function, +Config)
    lambda_config/2,                % lambda_config(?Function, ?Config)
    generate_lambda_function/3,     % generate_lambda_function(+Function, +Options, -Package)
    generate_lambda_deploy/3,       % generate_lambda_deploy(+Function, +Options, -Commands)
    generate_sam_template/3,        % generate_sam_template(+Function, +Options, -Template)

    % Google Cloud Functions
    declare_gcf_config/2,           % declare_gcf_config(+Function, +Config)
    gcf_config/2,                   % gcf_config(?Function, ?Config)
    generate_gcf_deploy/3,          % generate_gcf_deploy(+Function, +Options, -Commands)

    % Azure Functions
    declare_azure_func_config/2,    % declare_azure_func_config(+Function, +Config)
    azure_func_config/2,            % azure_func_config(?Function, ?Config)
    generate_azure_func_deploy/3,   % generate_azure_func_deploy(+Function, +Options, -Commands)

    % API Gateway integration
    declare_api_gateway/2,          % declare_api_gateway(+Name, +Config)
    api_gateway_config/2,           % api_gateway_config(?Name, ?Config)
    generate_api_gateway_config/3,  % generate_api_gateway_config(+Name, +Options, -Config)
    generate_openapi_spec/3,        % generate_openapi_spec(+Gateway, +Options, -Spec)

    % Unified serverless deployment
    deploy_function/3,              % deploy_function(+Function, +Options, -Result)
    invoke_function/4,              % invoke_function(+Function, +Payload, +Options, -Result)
    function_logs/3                 % function_logs(+Function, +Options, -Logs)
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

% Phase 7a dynamic storage
:- dynamic docker_config_db/2.          % docker_config_db(Service, Config)
:- dynamic compose_config_db/2.         % compose_config_db(Project, Config)
:- dynamic k8s_config_db/2.             % k8s_config_db(Service, Config)
:- dynamic registry_config_db/2.        % registry_config_db(Name, Config)

% Phase 7b dynamic storage
:- dynamic secret_source_db/2.          % secret_source_db(Name, Config)
:- dynamic vault_config_db/2.           % vault_config_db(Name, Config)
:- dynamic aws_secrets_config_db/2.     % aws_secrets_config_db(Name, Config)
:- dynamic azure_keyvault_config_db/2.  % azure_keyvault_config_db(Name, Config)
:- dynamic gcp_secrets_config_db/2.     % gcp_secrets_config_db(Name, Config)
:- dynamic service_secrets_db/2.        % service_secrets_db(Service, Secrets)
:- dynamic secret_rotation_db/3.        % secret_rotation_db(Service, Secret, Config)

% Phase 7c dynamic storage - Multi-Region
:- dynamic region_config_db/2.          % region_config_db(Name, Config)
:- dynamic service_regions_db/2.        % service_regions_db(Service, RegionConfig)
:- dynamic failover_policy_db/2.        % failover_policy_db(Service, Policy)
:- dynamic region_status_db/4.          % region_status_db(Service, Region, Status, Timestamp)
:- dynamic traffic_policy_db/2.         % traffic_policy_db(Service, Policy)

% Phase 7c dynamic storage - Cloud Functions
:- dynamic lambda_config_db/2.          % lambda_config_db(Function, Config)
:- dynamic gcf_config_db/2.             % gcf_config_db(Function, Config)
:- dynamic azure_func_config_db/2.      % azure_func_config_db(Function, Config)
:- dynamic api_gateway_config_db/2.     % api_gateway_config_db(Name, Config)

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

%% ============================================
%% Phase 7a: Container Deployment [EXPERIMENTAL]
%% ============================================
%%
%% WARNING: All Phase 7a predicates are EXPERIMENTAL.
%%
%% Current test coverage:
%% - Unit tests verify code generation (string/structure output)
%% - NO integration tests with actual Docker/Kubernetes
%% - Generated commands are NOT executed, only returned
%%
%% Before production use, verify:
%% - Generated Dockerfiles build successfully
%% - Generated K8s manifests pass: kubectl apply --dry-run=client
%% - Registry authentication works in your environment
%% - Commands execute correctly on your target systems
%%
%% ============================================

%% --------------------------------------------
%% Docker Configuration [EXPERIMENTAL]
%% --------------------------------------------

%% declare_docker_config(+Service, +Config)
%  Configure Docker settings for a service.
%
%  Config options:
%    - base_image(Image)     : Base Docker image (default: auto-detected)
%    - registry(Name)        : Registry to push to
%    - image_name(Name)      : Image name (default: service name)
%    - tag(Tag)              : Image tag (default: latest)
%    - build_args(Args)      : Build arguments (list of name-value pairs)
%    - labels(Labels)        : Docker labels
%    - expose(Ports)         : Ports to expose
%    - env(Vars)             : Environment variables
%    - volumes(Vols)         : Volume mounts
%    - healthcheck(HC)       : Health check configuration
%    - multi_stage(Bool)     : Use multi-stage build
%    - workdir(Dir)          : Working directory
%    - user(User)            : Run as user
%    - entrypoint(EP)        : Container entrypoint
%    - cmd(Cmd)              : Default command
%
declare_docker_config(Service, Config) :-
    retractall(docker_config_db(Service, _)),
    assertz(docker_config_db(Service, Config)).

%% docker_config(?Service, ?Config)
%  Query Docker configuration.
%
docker_config(Service, Config) :-
    docker_config_db(Service, Config).

%% --------------------------------------------
%% Dockerfile Generation
%% --------------------------------------------

%% generate_dockerfile(+Service, +Options, -Dockerfile)
%  Generate a Dockerfile for the service.
%
%  Options:
%    - optimize(Bool)        : Enable build optimizations
%    - no_cache_layers(List) : Layers to not cache
%
generate_dockerfile(Service, Options, Dockerfile) :-
    service_config(Service, ServiceConfig),
    docker_config(Service, DockerConfig),
    option_or_default(target, ServiceConfig, python, Target),
    generate_dockerfile_for_target(Target, Service, ServiceConfig, DockerConfig, Options, Dockerfile).

%% generate_dockerfile_for_target(+Target, +Service, +ServiceConfig, +DockerConfig, +Options, -Dockerfile)
%  Generate target-specific Dockerfile.
%
generate_dockerfile_for_target(python, Service, ServiceConfig, DockerConfig, _Options, Dockerfile) :-
    option_or_default(base_image, DockerConfig, 'python:3.11-slim', BaseImage),
    option_or_default(workdir, DockerConfig, '/app', WorkDir),
    option_or_default(entry_point, ServiceConfig, 'main.py', EntryPoint),
    option_or_default(port, ServiceConfig, 8080, Port),
    option_or_default(env, DockerConfig, [], EnvVars),
    option_or_default(healthcheck, DockerConfig, none, HealthCheck),
    option_or_default(user, DockerConfig, none, User),

    % Build environment lines
    format_env_lines(EnvVars, EnvLines),

    % Build healthcheck line
    format_healthcheck(HealthCheck, Port, HealthCheckLine),

    % Build user line
    format_user_line(User, UserLine),

    format(atom(Dockerfile),
'# Generated by UnifyWeaver - Phase 7a Container Deployment
# Service: ~w

FROM ~w

WORKDIR ~w

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

~w~w~w
EXPOSE ~w

CMD ["python", "~w"]
', [Service, BaseImage, WorkDir, EnvLines, UserLine, HealthCheckLine, Port, EntryPoint]).

generate_dockerfile_for_target(go, Service, ServiceConfig, DockerConfig, Options, Dockerfile) :-
    option_or_default(base_image, DockerConfig, 'golang:1.21-alpine', BuildImage),
    option_or_default(runtime_image, DockerConfig, 'alpine:latest', RuntimeImage),
    option_or_default(workdir, DockerConfig, '/app', WorkDir),
    option_or_default(entry_point, ServiceConfig, 'main.go', _EntryPoint),
    option_or_default(port, ServiceConfig, 8080, Port),
    option_or_default(multi_stage, DockerConfig, true, MultiStage),
    option_or_default(env, DockerConfig, [], EnvVars),
    atom_string(Service, ServiceStr),

    format_env_lines(EnvVars, EnvLines),

    (   MultiStage == true, \+ member(single_stage, Options)
    ->  format(atom(Dockerfile),
'# Generated by UnifyWeaver - Phase 7a Container Deployment
# Service: ~w (Go multi-stage build)

# Build stage
FROM ~w AS builder

WORKDIR ~w

# Download dependencies
COPY go.mod go.sum ./
RUN go mod download

# Build application
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o /~w .

# Runtime stage
FROM ~w

RUN apk --no-cache add ca-certificates

WORKDIR /root/

COPY --from=builder /~w .

~w
EXPOSE ~w

CMD ["./~w"]
', [Service, BuildImage, WorkDir, ServiceStr, RuntimeImage, ServiceStr, EnvLines, Port, ServiceStr])
    ;   format(atom(Dockerfile),
'# Generated by UnifyWeaver - Phase 7a Container Deployment
# Service: ~w (Go single-stage build)

FROM ~w

WORKDIR ~w

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o /~w .

~w
EXPOSE ~w

CMD ["/~w"]
', [Service, BuildImage, WorkDir, ServiceStr, EnvLines, Port, ServiceStr])
    ).

generate_dockerfile_for_target(rust, Service, ServiceConfig, DockerConfig, _Options, Dockerfile) :-
    option_or_default(base_image, DockerConfig, 'rust:1.73-slim', BuildImage),
    option_or_default(runtime_image, DockerConfig, 'debian:bookworm-slim', RuntimeImage),
    option_or_default(workdir, DockerConfig, '/app', WorkDir),
    option_or_default(port, ServiceConfig, 8080, Port),
    option_or_default(env, DockerConfig, [], EnvVars),
    atom_string(Service, ServiceStr),

    format_env_lines(EnvVars, EnvLines),

    format(atom(Dockerfile),
'# Generated by UnifyWeaver - Phase 7a Container Deployment
# Service: ~w (Rust multi-stage build)

# Build stage
FROM ~w AS builder

WORKDIR ~w

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Create dummy main to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# Build actual application
COPY src ./src
RUN touch src/main.rs && cargo build --release

# Runtime stage
FROM ~w

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder ~w/target/release/~w /usr/local/bin/

~w
EXPOSE ~w

CMD ["~w"]
', [Service, BuildImage, WorkDir, RuntimeImage, WorkDir, ServiceStr, EnvLines, Port, ServiceStr]).

generate_dockerfile_for_target(nodejs, Service, ServiceConfig, DockerConfig, _Options, Dockerfile) :-
    option_or_default(base_image, DockerConfig, 'node:20-alpine', BaseImage),
    option_or_default(workdir, DockerConfig, '/app', WorkDir),
    option_or_default(entry_point, ServiceConfig, 'index.js', EntryPoint),
    option_or_default(port, ServiceConfig, 3000, Port),
    option_or_default(env, DockerConfig, [], EnvVars),

    format_env_lines(EnvVars, EnvLines),

    format(atom(Dockerfile),
'# Generated by UnifyWeaver - Phase 7a Container Deployment
# Service: ~w (Node.js)

FROM ~w

WORKDIR ~w

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy application
COPY . .

~w
EXPOSE ~w

CMD ["node", "~w"]
', [Service, BaseImage, WorkDir, EnvLines, Port, EntryPoint]).

generate_dockerfile_for_target(csharp, Service, ServiceConfig, DockerConfig, _Options, Dockerfile) :-
    option_or_default(base_image, DockerConfig, 'mcr.microsoft.com/dotnet/sdk:8.0', BuildImage),
    option_or_default(runtime_image, DockerConfig, 'mcr.microsoft.com/dotnet/aspnet:8.0', RuntimeImage),
    option_or_default(workdir, DockerConfig, '/app', WorkDir),
    option_or_default(port, ServiceConfig, 8080, Port),
    option_or_default(env, DockerConfig, [], EnvVars),
    atom_string(Service, ServiceStr),

    format_env_lines(EnvVars, EnvLines),

    format(atom(Dockerfile),
'# Generated by UnifyWeaver - Phase 7a Container Deployment
# Service: ~w (C# .NET)

# Build stage
FROM ~w AS build

WORKDIR /src

COPY *.csproj ./
RUN dotnet restore

COPY . .
RUN dotnet publish -c Release -o /app/publish

# Runtime stage
FROM ~w

WORKDIR ~w

COPY --from=build /app/publish .

~w
EXPOSE ~w

ENTRYPOINT ["dotnet", "~w.dll"]
', [Service, BuildImage, RuntimeImage, WorkDir, EnvLines, Port, ServiceStr]).

% Fallback for unknown targets
generate_dockerfile_for_target(Target, Service, _ServiceConfig, DockerConfig, _Options, Dockerfile) :-
    option_or_default(base_image, DockerConfig, 'alpine:latest', BaseImage),
    format(atom(Dockerfile),
'# Generated by UnifyWeaver - Phase 7a Container Deployment
# Service: ~w (Target: ~w)
# WARNING: Unknown target, using generic template

FROM ~w

WORKDIR /app

COPY . .

# TODO: Add build/run commands for target: ~w
', [Service, Target, BaseImage, Target]).

%% format_env_lines(+EnvVars, -Lines)
%  Format environment variables for Dockerfile.
%
format_env_lines([], '').
format_env_lines(EnvVars, Lines) :-
    EnvVars \= [],
    maplist(format_env_var, EnvVars, EnvLines),
    atomic_list_concat(EnvLines, '\n', Lines0),
    format(atom(Lines), '~w\n', [Lines0]).

format_env_var(Name-Value, Line) :-
    format(atom(Line), 'ENV ~w=~w', [Name, Value]).
format_env_var(Name=Value, Line) :-
    format(atom(Line), 'ENV ~w=~w', [Name, Value]).

%% format_healthcheck(+HealthCheck, +Port, -Line)
%  Format health check for Dockerfile.
%
format_healthcheck(none, _, '').
format_healthcheck(http(Path), Port, Line) :-
    format(atom(Line), 'HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \\\n  CMD wget --quiet --tries=1 --spider http://localhost:~w~w || exit 1\n', [Port, Path]).
format_healthcheck(tcp, Port, Line) :-
    format(atom(Line), 'HEALTHCHECK --interval=30s --timeout=5s --retries=3 \\\n  CMD nc -z localhost ~w || exit 1\n', [Port]).
format_healthcheck(cmd(Cmd), _, Line) :-
    format(atom(Line), 'HEALTHCHECK --interval=30s --timeout=5s --retries=3 \\\n  CMD ~w\n', [Cmd]).

%% format_user_line(+User, -Line)
%  Format user line for Dockerfile.
%
format_user_line(none, '').
format_user_line(User, Line) :-
    User \= none,
    format(atom(Line), 'USER ~w\n', [User]).

%% generate_dockerignore(+Service, +Options, -Content)
%  Generate .dockerignore file content.
%
generate_dockerignore(Service, _Options, Content) :-
    service_config(Service, ServiceConfig),
    option_or_default(target, ServiceConfig, python, Target),
    generate_dockerignore_for_target(Target, Content).

generate_dockerignore_for_target(python, Content) :-
    Content = '# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.venv/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp

# Git
.git/
.gitignore

# Docker
Dockerfile
.dockerignore

# Tests
tests/
*_test.py
test_*.py
'.

generate_dockerignore_for_target(go, Content) :-
    Content = '# Go
*.exe
*.exe~
*.dll
*.so
*.dylib
*.test
*.out
vendor/
go.work

# IDE
.idea/
.vscode/
*.swp

# Git
.git/
.gitignore

# Docker
Dockerfile
.dockerignore

# Tests
*_test.go
'.

generate_dockerignore_for_target(rust, Content) :-
    Content = '# Rust
target/
**/*.rs.bk
Cargo.lock

# IDE
.idea/
.vscode/
*.swp

# Git
.git/
.gitignore

# Docker
Dockerfile
.dockerignore
'.

generate_dockerignore_for_target(nodejs, Content) :-
    Content = '# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.yarn

# IDE
.idea/
.vscode/
*.swp

# Git
.git/
.gitignore

# Docker
Dockerfile
.dockerignore

# Tests
test/
tests/
*.test.js
*.spec.js
__tests__/
'.

generate_dockerignore_for_target(csharp, Content) :-
    Content = '# .NET
bin/
obj/
*.user
*.suo
.vs/

# IDE
.idea/
.vscode/
*.swp

# Git
.git/
.gitignore

# Docker
Dockerfile
.dockerignore
'.

generate_dockerignore_for_target(_, Content) :-
    Content = '# Generic
.git/
.gitignore
Dockerfile
.dockerignore
.idea/
.vscode/
*.swp
'.

%% --------------------------------------------
%% Docker Operations
%% --------------------------------------------

%% docker_image_tag(+Service, -Tag)
%  Get the full Docker image tag for a service.
%
docker_image_tag(Service, Tag) :-
    docker_config(Service, Config),
    option_or_default(registry, Config, none, Registry),
    option_or_default(image_name, Config, Service, ImageName),
    option_or_default(tag, Config, latest, ImageTag),
    (   Registry == none
    ->  format(atom(Tag), '~w:~w', [ImageName, ImageTag])
    ;   registry_config(Registry, RegConfig),
        option_or_default(url, RegConfig, Registry, RegURL),
        option_or_default(image_prefix, RegConfig, '', Prefix),
        format(atom(Tag), '~w/~w~w:~w', [RegURL, Prefix, ImageName, ImageTag])
    ).

%% build_docker_image(+Service, +Options, -Result)
%  Build Docker image for a service.
%
%  Options:
%    - no_cache(Bool)       : Build without cache
%    - platform(P)          : Target platform (e.g., linux/amd64)
%    - build_args(Args)     : Additional build arguments
%
build_docker_image(Service, Options, Result) :-
    docker_image_tag(Service, Tag),
    option_or_default(no_cache, Options, false, NoCache),
    option_or_default(platform, Options, none, Platform),
    option_or_default(build_args, Options, [], BuildArgs),

    % Build command parts
    (NoCache == true -> NoCacheFlag = '--no-cache ' ; NoCacheFlag = ''),
    (Platform \= none -> format(atom(PlatformFlag), '--platform ~w ', [Platform]) ; PlatformFlag = ''),
    format_build_args(BuildArgs, BuildArgsStr),

    format(atom(Cmd), 'docker build ~w~w~w-t ~w .', [NoCacheFlag, PlatformFlag, BuildArgsStr, Tag]),

    Result = build_command(Cmd, Tag).

format_build_args([], '').
format_build_args(Args, Str) :-
    Args \= [],
    maplist(format_build_arg, Args, ArgStrs),
    atomic_list_concat(ArgStrs, ' ', Str0),
    format(atom(Str), '~w ', [Str0]).

format_build_arg(Name-Value, Str) :-
    format(atom(Str), '--build-arg ~w=~w', [Name, Value]).
format_build_arg(Name=Value, Str) :-
    format(atom(Str), '--build-arg ~w=~w', [Name, Value]).

%% push_docker_image(+Service, +Options, -Result)
%  Push Docker image to registry.
%
push_docker_image(Service, _Options, Result) :-
    docker_image_tag(Service, Tag),
    format(atom(Cmd), 'docker push ~w', [Tag]),
    Result = push_command(Cmd, Tag).

%% --------------------------------------------
%% Docker Compose
%% --------------------------------------------

%% declare_compose_config(+Project, +Config)
%  Configure Docker Compose project.
%
%  Config options:
%    - services(List)       : List of service names
%    - networks(List)       : Network configurations
%    - volumes(List)        : Volume configurations
%    - version(V)           : Compose file version (default: '3.8')
%
declare_compose_config(Project, Config) :-
    retractall(compose_config_db(Project, _)),
    assertz(compose_config_db(Project, Config)).

%% compose_config(?Project, ?Config)
%  Query Docker Compose configuration.
%
compose_config(Project, Config) :-
    compose_config_db(Project, Config).

%% generate_docker_compose(+Project, +Options, -ComposeYaml)
%  Generate docker-compose.yml content.
%
generate_docker_compose(Project, _Options, ComposeYaml) :-
    compose_config(Project, Config),
    option_or_default(services, Config, [], ServiceNames),
    option_or_default(networks, Config, [default], Networks),
    option_or_default(volumes, Config, [], Volumes),
    option_or_default(version, Config, '3.8', Version),

    % Generate services section
    maplist(generate_compose_service, ServiceNames, ServiceYamls),
    atomic_list_concat(ServiceYamls, '\n', ServicesSection),

    % Generate networks section
    generate_compose_networks(Networks, NetworksSection),

    % Generate volumes section
    generate_compose_volumes(Volumes, VolumesSection),

    format(atom(ComposeYaml),
'# Generated by UnifyWeaver - Phase 7a Container Deployment
# Project: ~w

version: "~w"

services:
~w
~w~w', [Project, Version, ServicesSection, NetworksSection, VolumesSection]).

%% generate_compose_service(+ServiceName, -ServiceYaml)
%  Generate service entry for docker-compose.yml.
%
generate_compose_service(ServiceName, ServiceYaml) :-
    (   service_config(ServiceName, ServiceConfig),
        docker_config(ServiceName, DockerConfig)
    ->  option_or_default(port, ServiceConfig, 8080, Port),
        option_or_default(env, DockerConfig, [], EnvVars),
        option_or_default(volumes, DockerConfig, [], Volumes),
        option_or_default(depends_on, DockerConfig, [], DependsOn),
        option_or_default(restart, DockerConfig, 'unless-stopped', Restart),

        % Format environment
        format_compose_env(EnvVars, EnvSection),

        % Format volumes
        format_compose_volumes(Volumes, VolSection),

        % Format depends_on
        format_compose_depends(DependsOn, DependsSection),

        docker_image_tag(ServiceName, ImageTag),

        format(atom(ServiceYaml),
'  ~w:
    image: ~w
    ports:
      - "~w:~w"
    restart: ~w~w~w~w
', [ServiceName, ImageTag, Port, Port, Restart, EnvSection, VolSection, DependsSection])
    ;   % Service not fully configured, use defaults
        format(atom(ServiceYaml),
'  ~w:
    build: ./~w
    restart: unless-stopped
', [ServiceName, ServiceName])
    ).

format_compose_env([], '').
format_compose_env(EnvVars, Section) :-
    EnvVars \= [],
    maplist(format_compose_env_var, EnvVars, Lines),
    atomic_list_concat(Lines, '\n', LinesStr),
    format(atom(Section), '\n    environment:\n~w', [LinesStr]).

format_compose_env_var(Name-Value, Line) :-
    format(atom(Line), '      - ~w=~w', [Name, Value]).
format_compose_env_var(Name=Value, Line) :-
    format(atom(Line), '      - ~w=~w', [Name, Value]).

format_compose_volumes([], '').
format_compose_volumes(Volumes, Section) :-
    Volumes \= [],
    maplist(format_compose_volume, Volumes, Lines),
    atomic_list_concat(Lines, '\n', LinesStr),
    format(atom(Section), '\n    volumes:\n~w', [LinesStr]).

format_compose_volume(Source:Target, Line) :-
    format(atom(Line), '      - ~w:~w', [Source, Target]).
format_compose_volume(Vol, Line) :-
    \+ (Vol = _:_),
    format(atom(Line), '      - ~w', [Vol]).

format_compose_depends([], '').
format_compose_depends(DependsOn, Section) :-
    DependsOn \= [],
    maplist(format_compose_depend, DependsOn, Lines),
    atomic_list_concat(Lines, '\n', LinesStr),
    format(atom(Section), '\n    depends_on:\n~w', [LinesStr]).

format_compose_depend(Service, Line) :-
    format(atom(Line), '      - ~w', [Service]).

generate_compose_networks([], '').
generate_compose_networks([default], '').
generate_compose_networks(Networks, Section) :-
    Networks \= [],
    Networks \= [default],
    maplist(format_compose_network, Networks, Lines),
    atomic_list_concat(Lines, '\n', LinesStr),
    format(atom(Section), '\nnetworks:\n~w\n', [LinesStr]).

format_compose_network(Name, Line) :-
    format(atom(Line), '  ~w:', [Name]).

generate_compose_volumes([], '').
generate_compose_volumes(Volumes, Section) :-
    Volumes \= [],
    maplist(format_compose_named_volume, Volumes, Lines),
    atomic_list_concat(Lines, '\n', LinesStr),
    format(atom(Section), '\nvolumes:\n~w\n', [LinesStr]).

format_compose_named_volume(Name, Line) :-
    format(atom(Line), '  ~w:', [Name]).

%% --------------------------------------------
%% Kubernetes Deployment
%% --------------------------------------------

%% declare_k8s_config(+Service, +Config)
%  Configure Kubernetes deployment settings.
%
%  Config options:
%    - namespace(NS)        : Kubernetes namespace (default: 'default')
%    - replicas(N)          : Number of replicas (default: 1)
%    - resources(Res)       : Resource requests/limits
%    - env(Vars)            : Environment variables
%    - env_from(Sources)    : ConfigMap/Secret references
%    - service_type(Type)   : ClusterIP | NodePort | LoadBalancer
%    - node_port(Port)      : For NodePort services
%    - ingress(Config)      : Ingress configuration
%    - labels(Labels)       : Additional labels
%    - annotations(Ann)     : Annotations
%    - liveness_probe(P)    : Liveness probe config
%    - readiness_probe(P)   : Readiness probe config
%    - image_pull_policy(P) : Always | IfNotPresent | Never
%    - image_pull_secrets(S): Image pull secret names
%
declare_k8s_config(Service, Config) :-
    retractall(k8s_config_db(Service, _)),
    assertz(k8s_config_db(Service, Config)).

%% k8s_config(?Service, ?Config)
%  Query Kubernetes configuration.
%
k8s_config(Service, Config) :-
    k8s_config_db(Service, Config).

%% generate_k8s_deployment(+Service, +Options, -Manifest)
%  Generate Kubernetes Deployment manifest.
%
generate_k8s_deployment(Service, _Options, Manifest) :-
    k8s_config(Service, Config),
    service_config(Service, ServiceConfig),
    docker_image_tag(Service, ImageTag),

    option_or_default(namespace, Config, 'default', Namespace),
    option_or_default(replicas, Config, 1, Replicas),
    option_or_default(port, ServiceConfig, 8080, Port),
    option_or_default(resources, Config, default, Resources),
    option_or_default(env, Config, [], EnvVars),
    option_or_default(labels, Config, [], ExtraLabels),
    option_or_default(image_pull_policy, Config, 'IfNotPresent', ImagePullPolicy),
    option_or_default(liveness_probe, Config, none, LivenessProbe),
    option_or_default(readiness_probe, Config, none, ReadinessProbe),

    atom_string(Service, ServiceStr),

    % Format resources
    format_k8s_resources(Resources, ResourcesSection),

    % Format env vars
    format_k8s_env(EnvVars, EnvSection),

    % Format labels
    format_k8s_labels(ExtraLabels, ServiceStr, LabelsSection),

    % Format probes
    format_k8s_probe(liveness, LivenessProbe, Port, LivenessSection),
    format_k8s_probe(readiness, ReadinessProbe, Port, ReadinessSection),

    format(atom(Manifest),
'# Generated by UnifyWeaver - Phase 7a Container Deployment
# Kubernetes Deployment for ~w
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ~w
  namespace: ~w
  labels:
    app: ~w
~w
spec:
  replicas: ~w
  selector:
    matchLabels:
      app: ~w
  template:
    metadata:
      labels:
        app: ~w
~w
    spec:
      containers:
      - name: ~w
        image: ~w
        imagePullPolicy: ~w
        ports:
        - containerPort: ~w
~w~w~w~w', [Service, ServiceStr, Namespace, ServiceStr, LabelsSection,
             Replicas, ServiceStr, ServiceStr, LabelsSection,
             ServiceStr, ImageTag, ImagePullPolicy, Port,
             EnvSection, ResourcesSection, LivenessSection, ReadinessSection]).

format_k8s_resources(default, Section) :-
    Section = '        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "500m"
'.
format_k8s_resources(none, '').
format_k8s_resources(resources(Requests, Limits), Section) :-
    format_k8s_resource_spec(Requests, ReqStr),
    format_k8s_resource_spec(Limits, LimStr),
    format(atom(Section), '        resources:
          requests:
~w
          limits:
~w
', [ReqStr, LimStr]).

format_k8s_resource_spec([], '').
format_k8s_resource_spec(Specs, Str) :-
    Specs \= [],
    maplist(format_k8s_resource_item, Specs, Lines),
    atomic_list_concat(Lines, '\n', Str).

format_k8s_resource_item(memory-Val, Line) :-
    format(atom(Line), '            memory: "~w"', [Val]).
format_k8s_resource_item(cpu-Val, Line) :-
    format(atom(Line), '            cpu: "~w"', [Val]).

format_k8s_env([], '').
format_k8s_env(EnvVars, Section) :-
    EnvVars \= [],
    maplist(format_k8s_env_var, EnvVars, Lines),
    atomic_list_concat(Lines, '\n', LinesStr),
    format(atom(Section), '        env:
~w
', [LinesStr]).

format_k8s_env_var(Name-Value, Line) :-
    format(atom(Line), '        - name: ~w
          value: "~w"', [Name, Value]).
format_k8s_env_var(Name=Value, Line) :-
    format(atom(Line), '        - name: ~w
          value: "~w"', [Name, Value]).
format_k8s_env_var(secret_ref(Name, Key), Line) :-
    format(atom(Line), '        - name: ~w
          valueFrom:
            secretKeyRef:
              name: ~w
              key: ~w', [Key, Name, Key]).
format_k8s_env_var(configmap_ref(Name, Key), Line) :-
    format(atom(Line), '        - name: ~w
          valueFrom:
            configMapKeyRef:
              name: ~w
              key: ~w', [Key, Name, Key]).

format_k8s_labels([], _, '').
format_k8s_labels(Labels, _, Section) :-
    Labels \= [],
    maplist(format_k8s_label, Labels, Lines),
    atomic_list_concat(Lines, '\n', LinesStr),
    format(atom(Section), '~w', [LinesStr]).

format_k8s_label(Name-Value, Line) :-
    format(atom(Line), '    ~w: ~w', [Name, Value]).

format_k8s_probe(_, none, _, '').
format_k8s_probe(Type, http(Path), Port, Section) :-
    (Type == liveness -> TypeName = 'livenessProbe' ; TypeName = 'readinessProbe'),
    format(atom(Section), '        ~w:
          httpGet:
            path: ~w
            port: ~w
          initialDelaySeconds: 10
          periodSeconds: 10
', [TypeName, Path, Port]).
format_k8s_probe(Type, tcp, Port, Section) :-
    (Type == liveness -> TypeName = 'livenessProbe' ; TypeName = 'readinessProbe'),
    format(atom(Section), '        ~w:
          tcpSocket:
            port: ~w
          initialDelaySeconds: 10
          periodSeconds: 10
', [TypeName, Port]).
format_k8s_probe(Type, exec(Cmd), _, Section) :-
    (Type == liveness -> TypeName = 'livenessProbe' ; TypeName = 'readinessProbe'),
    format(atom(Section), '        ~w:
          exec:
            command:
            - ~w
          initialDelaySeconds: 10
          periodSeconds: 10
', [TypeName, Cmd]).

%% generate_k8s_service(+Service, +Options, -Manifest)
%  Generate Kubernetes Service manifest.
%
generate_k8s_service(Service, _Options, Manifest) :-
    k8s_config(Service, Config),
    service_config(Service, ServiceConfig),

    option_or_default(namespace, Config, 'default', Namespace),
    option_or_default(port, ServiceConfig, 8080, Port),
    option_or_default(service_type, Config, 'ClusterIP', ServiceType),
    option_or_default(node_port, Config, none, NodePort),

    atom_string(Service, ServiceStr),

    % Format node port if applicable
    (   NodePort \= none, ServiceType == 'NodePort'
    ->  format(atom(NodePortLine), '    nodePort: ~w\n', [NodePort])
    ;   NodePortLine = ''
    ),

    format(atom(Manifest),
'# Generated by UnifyWeaver - Phase 7a Container Deployment
# Kubernetes Service for ~w
apiVersion: v1
kind: Service
metadata:
  name: ~w
  namespace: ~w
spec:
  type: ~w
  selector:
    app: ~w
  ports:
  - port: ~w
    targetPort: ~w
~w', [Service, ServiceStr, Namespace, ServiceType, ServiceStr, Port, Port, NodePortLine]).

%% generate_k8s_configmap(+Service, +Options, -Manifest)
%  Generate Kubernetes ConfigMap manifest.
%
generate_k8s_configmap(Service, Options, Manifest) :-
    k8s_config(Service, Config),

    option_or_default(namespace, Config, 'default', Namespace),
    option_or_default(config_data, Options, [], ConfigData),

    atom_string(Service, ServiceStr),
    format(atom(ConfigMapName), '~w-config', [ServiceStr]),

    % Format config data
    format_k8s_configmap_data(ConfigData, DataSection),

    format(atom(Manifest),
'# Generated by UnifyWeaver - Phase 7a Container Deployment
# Kubernetes ConfigMap for ~w
apiVersion: v1
kind: ConfigMap
metadata:
  name: ~w
  namespace: ~w
data:
~w', [Service, ConfigMapName, Namespace, DataSection]).

format_k8s_configmap_data([], '  # No data configured\n').
format_k8s_configmap_data(Data, Section) :-
    Data \= [],
    maplist(format_k8s_configmap_item, Data, Lines),
    atomic_list_concat(Lines, '\n', Section).

format_k8s_configmap_item(Key-Value, Line) :-
    format(atom(Line), '  ~w: "~w"', [Key, Value]).

%% generate_k8s_ingress(+Service, +Options, -Manifest)
%  Generate Kubernetes Ingress manifest.
%
generate_k8s_ingress(Service, _Options, Manifest) :-
    k8s_config(Service, Config),
    service_config(Service, ServiceConfig),

    option_or_default(namespace, Config, 'default', Namespace),
    option_or_default(port, ServiceConfig, 8080, Port),
    option_or_default(ingress, Config, [], IngressConfig),
    option_or_default(host, IngressConfig, none, Host),
    option_or_default(path, IngressConfig, '/', Path),
    option_or_default(tls, IngressConfig, false, TLS),
    option_or_default(ingress_class, IngressConfig, 'nginx', IngressClass),

    atom_string(Service, ServiceStr),

    % Format TLS section
    (   TLS == true, Host \= none
    ->  format(atom(TLSSection), '  tls:
  - hosts:
    - ~w
    secretName: ~w-tls
', [Host, ServiceStr])
    ;   TLSSection = ''
    ),

    % Format host rule
    (   Host \= none
    ->  format(atom(HostLine), '    - host: ~w\n      http:', [Host])
    ;   HostLine = '    - http:'
    ),

    format(atom(Manifest),
'# Generated by UnifyWeaver - Phase 7a Container Deployment
# Kubernetes Ingress for ~w
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ~w
  namespace: ~w
  annotations:
    kubernetes.io/ingress.class: ~w
spec:
~w  rules:
~w
        paths:
        - path: ~w
          pathType: Prefix
          backend:
            service:
              name: ~w
              port:
                number: ~w
', [Service, ServiceStr, Namespace, IngressClass, TLSSection, HostLine, Path, ServiceStr, Port]).

%% generate_helm_chart(+Service, +Options, -Chart)
%  Generate Helm chart structure.
%
generate_helm_chart(Service, Options, Chart) :-
    atom_string(Service, ServiceStr),

    % Generate Chart.yaml
    option_or_default(version, Options, '0.1.0', ChartVersion),
    option_or_default(app_version, Options, '1.0.0', AppVersion),
    option_or_default(description, Options, 'A Helm chart for UnifyWeaver service', Description),

    format(atom(ChartYaml),
'apiVersion: v2
name: ~w
description: ~w
type: application
version: ~w
appVersion: "~w"
', [ServiceStr, Description, ChartVersion, AppVersion]),

    % Generate values.yaml
    k8s_config(Service, Config),
    service_config(Service, ServiceConfig),
    docker_image_tag(Service, ImageTag),

    option_or_default(replicas, Config, 1, Replicas),
    option_or_default(port, ServiceConfig, 8080, Port),
    option_or_default(service_type, Config, 'ClusterIP', ServiceType),

    format(atom(ValuesYaml),
'# Default values for ~w
replicaCount: ~w

image:
  repository: ~w
  pullPolicy: IfNotPresent
  tag: ""

service:
  type: ~w
  port: ~w

ingress:
  enabled: false
  className: ""
  annotations: {}
  hosts:
    - host: chart-example.local
      paths:
        - path: /
          pathType: Prefix
  tls: []

resources:
  limits:
    cpu: 500m
    memory: 256Mi
  requests:
    cpu: 100m
    memory: 128Mi

autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80

nodeSelector: {}
tolerations: []
affinity: {}
', [ServiceStr, Replicas, ImageTag, ServiceType, Port]),

    Chart = helm_chart{
        'Chart.yaml': ChartYaml,
        'values.yaml': ValuesYaml,
        name: ServiceStr
    }.

%% --------------------------------------------
%% Container Registry
%% --------------------------------------------

%% declare_registry(+Name, +Config)
%  Configure a container registry.
%
%  Config options:
%    - url(URL)             : Registry URL
%    - username(U)          : Username for authentication
%    - password_env(Var)    : Environment variable containing password
%    - image_prefix(P)      : Prefix for image names
%    - auth_method(M)       : token | basic | aws_ecr | gcp_gcr
%
declare_registry(Name, Config) :-
    retractall(registry_config_db(Name, _)),
    assertz(registry_config_db(Name, Config)).

%% registry_config(?Name, ?Config)
%  Query registry configuration.
%
registry_config(Name, Config) :-
    registry_config_db(Name, Config).

%% login_registry(+Name, -Result)
%  Generate login command for registry.
%
login_registry(Name, Result) :-
    registry_config(Name, Config),
    option_or_default(url, Config, Name, URL),
    option_or_default(auth_method, Config, basic, AuthMethod),
    option_or_default(username, Config, none, Username),
    option_or_default(password_env, Config, none, PasswordEnv),

    generate_login_command(AuthMethod, URL, Username, PasswordEnv, Cmd),
    Result = login_command(Name, Cmd).

generate_login_command(basic, URL, Username, PasswordEnv, Cmd) :-
    (   Username \= none, PasswordEnv \= none
    ->  format(atom(Cmd), 'echo $~w | docker login ~w -u ~w --password-stdin', [PasswordEnv, URL, Username])
    ;   format(atom(Cmd), 'docker login ~w', [URL])
    ).
generate_login_command(aws_ecr, URL, _, _, Cmd) :-
    format(atom(Cmd), 'aws ecr get-login-password | docker login --username AWS --password-stdin ~w', [URL]).
generate_login_command(gcp_gcr, URL, _, _, Cmd) :-
    format(atom(Cmd), 'gcloud auth configure-docker ~w', [URL]).
generate_login_command(token, URL, _, PasswordEnv, Cmd) :-
    format(atom(Cmd), 'echo $~w | docker login ~w --password-stdin', [PasswordEnv, URL]).

%% --------------------------------------------
%% Container Orchestration
%% --------------------------------------------

%% deploy_to_k8s(+Service, +Options, -Result)
%  Deploy service to Kubernetes.
%
%  Options:
%    - namespace(NS)        : Target namespace
%    - wait(Bool)           : Wait for rollout
%    - timeout(Secs)        : Rollout timeout
%
deploy_to_k8s(Service, Options, Result) :-
    atom_string(Service, ServiceStr),
    option_or_default(namespace, Options, 'default', Namespace),
    option_or_default(wait, Options, true, Wait),
    option_or_default(timeout, Options, 300, Timeout),

    % Generate manifests
    generate_k8s_deployment(Service, Options, DeploymentManifest),
    generate_k8s_service(Service, Options, ServiceManifest),

    % Build apply commands
    format(atom(ApplyDeployment), 'kubectl apply -f - <<EOF\n~w\nEOF', [DeploymentManifest]),
    format(atom(ApplyService), 'kubectl apply -f - <<EOF\n~w\nEOF', [ServiceManifest]),

    % Build wait command if needed
    (   Wait == true
    ->  format(atom(WaitCmd), 'kubectl rollout status deployment/~w -n ~w --timeout=~ws',
               [ServiceStr, Namespace, Timeout])
    ;   WaitCmd = ''
    ),

    Result = k8s_deploy{
        deployment_cmd: ApplyDeployment,
        service_cmd: ApplyService,
        wait_cmd: WaitCmd,
        service: Service,
        namespace: Namespace
    }.

%% scale_k8s_deployment(+Service, +Replicas, +Options, -Result)
%  Scale Kubernetes deployment.
%
scale_k8s_deployment(Service, Replicas, Options, Result) :-
    atom_string(Service, ServiceStr),
    option_or_default(namespace, Options, 'default', Namespace),

    format(atom(Cmd), 'kubectl scale deployment/~w --replicas=~w -n ~w',
           [ServiceStr, Replicas, Namespace]),

    Result = scale_command(Cmd, Service, Replicas).

%% rollout_status(+Service, +Options, -Status)
%  Check Kubernetes rollout status.
%
rollout_status(Service, Options, Status) :-
    atom_string(Service, ServiceStr),
    option_or_default(namespace, Options, 'default', Namespace),

    format(atom(Cmd), 'kubectl rollout status deployment/~w -n ~w',
           [ServiceStr, Namespace]),

    Status = rollout_status_command(Cmd, Service).

%% ============================================
%% Phase 7b: Secrets Management [EXPERIMENTAL]
%% ============================================
%%
%% WARNING: All Phase 7b predicates are EXPERIMENTAL.
%%
%% Current test coverage:
%% - Unit tests verify code generation (command strings, YAML output)
%% - NO integration tests with actual secret management services
%% - Generated commands are NOT executed, only returned
%%
%% Before production use, verify:
%% - Vault/AWS/Azure/GCP credentials are properly configured
%% - Generated commands execute correctly in your environment
%% - Secret rotation works as expected
%% - Kubernetes ExternalSecrets operator is installed (if using)
%%
%% ============================================

%% --------------------------------------------
%% Secret Source Declarations [EXPERIMENTAL]
%% --------------------------------------------

%% declare_secret_source(+Name, +Config)
%  Declare a generic secret source.
%
%  Config options:
%    - type(Type)           : vault | aws | azure | gcp | env | file
%    - priority(N)          : Priority for fallback (lower = higher priority)
%
declare_secret_source(Name, Config) :-
    retractall(secret_source_db(Name, _)),
    assertz(secret_source_db(Name, Config)).

%% secret_source_config(?Name, ?Config)
%  Query secret source configuration.
%
secret_source_config(Name, Config) :-
    secret_source_db(Name, Config).

%% --------------------------------------------
%% HashiCorp Vault [EXPERIMENTAL]
%% --------------------------------------------

%% declare_vault_config(+Name, +Config)
%  Configure HashiCorp Vault connection.
%
%  Config options:
%    - url(URL)             : Vault server URL
%    - auth_method(Method)  : token | approle | kubernetes | aws_iam
%    - token_env(Var)       : Environment variable for token
%    - role_id_env(Var)     : Environment variable for AppRole role_id
%    - secret_id_env(Var)   : Environment variable for AppRole secret_id
%    - k8s_role(Role)       : Kubernetes auth role
%    - namespace(NS)        : Vault namespace (enterprise)
%    - mount_path(Path)     : Secret engine mount path (default: secret)
%
declare_vault_config(Name, Config) :-
    retractall(vault_config_db(Name, _)),
    assertz(vault_config_db(Name, Config)).

%% vault_config(?Name, ?Config)
%  Query Vault configuration.
%
vault_config(Name, Config) :-
    vault_config_db(Name, Config).

%% generate_vault_read(+Source, +Path, +Options, -Command)
%  Generate command to read secret from Vault.
%
%  Options:
%    - field(F)             : Specific field to extract
%    - format(F)            : json | table | raw (default: json)
%    - version(V)           : KV v2 version number
%
generate_vault_read(Source, Path, Options, Command) :-
    vault_config(Source, Config),
    option_or_default(url, Config, 'http://127.0.0.1:8200', URL),
    option_or_default(auth_method, Config, token, AuthMethod),
    option_or_default(mount_path, Config, 'secret', MountPath),
    option_or_default(namespace, Config, none, Namespace),
    option_or_default(format, Options, json, Format),
    option_or_default(field, Options, none, Field),
    option_or_default(version, Options, none, Version),

    % Build auth part
    generate_vault_auth_env(AuthMethod, Config, AuthEnv),

    % Build namespace flag
    (Namespace \= none -> format(atom(NSFlag), '-namespace=~w ', [Namespace]) ; NSFlag = ''),

    % Build field flag
    (Field \= none -> format(atom(FieldFlag), '-field=~w ', [Field]) ; FieldFlag = ''),

    % Build version flag (KV v2)
    (Version \= none -> format(atom(VersionFlag), '-version=~w ', [Version]) ; VersionFlag = ''),

    % Full path with mount
    format(atom(FullPath), '~w/data/~w', [MountPath, Path]),

    format(atom(Command),
'~wVAULT_ADDR=~w vault kv get ~w~w~w-format=~w ~w',
           [AuthEnv, URL, NSFlag, FieldFlag, VersionFlag, Format, FullPath]).

%% generate_vault_auth_env(+Method, +Config, -EnvSetup)
%  Generate environment setup for Vault authentication.
%
generate_vault_auth_env(token, Config, EnvSetup) :-
    option_or_default(token_env, Config, 'VAULT_TOKEN', TokenEnv),
    format(atom(EnvSetup), 'VAULT_TOKEN=$~w ', [TokenEnv]).

generate_vault_auth_env(approle, Config, EnvSetup) :-
    option_or_default(role_id_env, Config, 'VAULT_ROLE_ID', RoleIdEnv),
    option_or_default(secret_id_env, Config, 'VAULT_SECRET_ID', SecretIdEnv),
    format(atom(EnvSetup),
'VAULT_TOKEN=$(vault write -field=token auth/approle/login role_id=$~w secret_id=$~w) ',
           [RoleIdEnv, SecretIdEnv]).

generate_vault_auth_env(kubernetes, Config, EnvSetup) :-
    option_or_default(k8s_role, Config, 'default', Role),
    format(atom(EnvSetup),
'VAULT_TOKEN=$(vault write -field=token auth/kubernetes/login role=~w jwt=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)) ',
           [Role]).

generate_vault_auth_env(aws_iam, _Config, EnvSetup) :-
    EnvSetup = 'VAULT_TOKEN=$(vault login -method=aws -field=token) '.

%% generate_vault_agent_config(+Service, +Options, -Config)
%  Generate Vault Agent configuration for automatic secret injection.
%
generate_vault_agent_config(Service, Options, Config) :-
    service_secrets(Service, Secrets),
    option_or_default(vault_source, Options, default_vault, VaultSource),
    vault_config(VaultSource, VaultConfig),
    option_or_default(url, VaultConfig, 'http://127.0.0.1:8200', URL),
    option_or_default(auth_method, VaultConfig, kubernetes, AuthMethod),
    option_or_default(k8s_role, VaultConfig, Service, Role),

    atom_string(Service, ServiceStr),

    % Generate template stanzas for each secret
    generate_vault_templates(Secrets, TemplateStanzas),

    % Generate auth config
    generate_vault_agent_auth(AuthMethod, Role, AuthStanza),

    format(atom(Config),
'# Vault Agent configuration for ~w
# Generated by UnifyWeaver - Phase 7b Secrets Management

vault {
  address = "~w"
}

auto_auth {
~w
}

template_config {
  static_secret_render_interval = "5m"
}

~w
', [ServiceStr, URL, AuthStanza, TemplateStanzas]).

generate_vault_agent_auth(kubernetes, Role, Stanza) :-
    format(atom(Stanza),
'  method "kubernetes" {
    mount_path = "auth/kubernetes"
    config = {
      role = "~w"
    }
  }

  sink "file" {
    config = {
      path = "/home/vault/.vault-token"
    }
  }', [Role]).

generate_vault_agent_auth(approle, _Role, Stanza) :-
    Stanza = '  method "approle" {
    mount_path = "auth/approle"
    config = {
      role_id_file_path = "/etc/vault/role-id"
      secret_id_file_path = "/etc/vault/secret-id"
    }
  }

  sink "file" {
    config = {
      path = "/home/vault/.vault-token"
    }
  }'.

generate_vault_agent_auth(token, _Role, Stanza) :-
    Stanza = '  method "token_file" {
    config = {
      token_file_path = "/etc/vault/token"
    }
  }'.

generate_vault_templates([], '').
generate_vault_templates([Secret|Rest], Stanzas) :-
    generate_vault_template(Secret, Stanza),
    generate_vault_templates(Rest, RestStanzas),
    format(atom(Stanzas), '~w\n~w', [Stanza, RestStanzas]).

generate_vault_template(secret(EnvVar, Path, Field), Stanza) :-
    format(atom(Stanza),
'template {
  destination = "/etc/secrets/~w"
  contents = <<EOF
{{ with secret "~w" }}{{ .Data.data.~w }}{{ end }}
EOF
}', [EnvVar, Path, Field]).

generate_vault_template(secret(EnvVar, Path), Stanza) :-
    format(atom(Stanza),
'template {
  destination = "/etc/secrets/~w"
  contents = <<EOF
{{ with secret "~w" }}{{ .Data.data | toJSON }}{{ end }}
EOF
}', [EnvVar, Path]).

%% --------------------------------------------
%% AWS Secrets Manager [EXPERIMENTAL]
%% --------------------------------------------

%% declare_aws_secrets_config(+Name, +Config)
%  Configure AWS Secrets Manager connection.
%
%  Config options:
%    - region(R)            : AWS region
%    - profile(P)           : AWS CLI profile
%    - role_arn(ARN)        : Role to assume
%    - endpoint_url(URL)    : Custom endpoint (for LocalStack, etc.)
%
declare_aws_secrets_config(Name, Config) :-
    retractall(aws_secrets_config_db(Name, _)),
    assertz(aws_secrets_config_db(Name, Config)).

%% aws_secrets_config(?Name, ?Config)
%  Query AWS Secrets Manager configuration.
%
aws_secrets_config(Name, Config) :-
    aws_secrets_config_db(Name, Config).

%% generate_aws_secret_read(+Source, +SecretId, +Options, -Command)
%  Generate command to read secret from AWS Secrets Manager.
%
%  Options:
%    - version_id(V)        : Specific version
%    - version_stage(S)     : AWSCURRENT | AWSPREVIOUS | custom
%    - key(K)               : JSON key to extract (uses jq)
%
generate_aws_secret_read(Source, SecretId, Options, Command) :-
    aws_secrets_config(Source, Config),
    option_or_default(region, Config, 'us-east-1', Region),
    option_or_default(profile, Config, none, Profile),
    option_or_default(endpoint_url, Config, none, Endpoint),
    option_or_default(version_id, Options, none, VersionId),
    option_or_default(version_stage, Options, none, VersionStage),
    option_or_default(key, Options, none, Key),

    % Build profile flag
    (Profile \= none -> format(atom(ProfileFlag), '--profile ~w ', [Profile]) ; ProfileFlag = ''),

    % Build endpoint flag
    (Endpoint \= none -> format(atom(EndpointFlag), '--endpoint-url ~w ', [Endpoint]) ; EndpointFlag = ''),

    % Build version flags
    (VersionId \= none -> format(atom(VersionIdFlag), '--version-id ~w ', [VersionId]) ; VersionIdFlag = ''),
    (VersionStage \= none -> format(atom(VersionStageFlag), '--version-stage ~w ', [VersionStage]) ; VersionStageFlag = ''),

    % Build jq filter for key extraction
    (Key \= none
    ->  format(atom(JqPart), ' | jq -r \'.SecretString | fromjson | .~w\'', [Key])
    ;   JqPart = ' | jq -r \'.SecretString\''
    ),

    format(atom(Command),
'aws secretsmanager get-secret-value ~w--region ~w ~w~w~w--secret-id ~w --query SecretString --output text~w',
           [ProfileFlag, Region, EndpointFlag, VersionIdFlag, VersionStageFlag, SecretId, JqPart]).

%% --------------------------------------------
%% Azure Key Vault [EXPERIMENTAL]
%% --------------------------------------------

%% declare_azure_keyvault_config(+Name, +Config)
%  Configure Azure Key Vault connection.
%
%  Config options:
%    - vault_name(Name)     : Key Vault name
%    - subscription(Sub)    : Azure subscription ID
%    - tenant_id(Tenant)    : Azure AD tenant ID
%    - use_msi(Bool)        : Use Managed Service Identity
%
declare_azure_keyvault_config(Name, Config) :-
    retractall(azure_keyvault_config_db(Name, _)),
    assertz(azure_keyvault_config_db(Name, Config)).

%% azure_keyvault_config(?Name, ?Config)
%  Query Azure Key Vault configuration.
%
azure_keyvault_config(Name, Config) :-
    azure_keyvault_config_db(Name, Config).

%% generate_azure_secret_read(+Source, +SecretName, +Options, -Command)
%  Generate command to read secret from Azure Key Vault.
%
%  Options:
%    - version(V)           : Specific secret version
%
generate_azure_secret_read(Source, SecretName, Options, Command) :-
    azure_keyvault_config(Source, Config),
    option_or_default(vault_name, Config, '', VaultName),
    option_or_default(subscription, Config, none, Subscription),
    option_or_default(version, Options, none, Version),

    % Build subscription flag
    (Subscription \= none -> format(atom(SubFlag), '--subscription ~w ', [Subscription]) ; SubFlag = ''),

    % Build version flag
    (Version \= none -> format(atom(VersionFlag), '--version ~w ', [Version]) ; VersionFlag = ''),

    format(atom(Command),
'az keyvault secret show ~w--vault-name ~w --name ~w ~w--query value -o tsv',
           [SubFlag, VaultName, SecretName, VersionFlag]).

%% --------------------------------------------
%% GCP Secret Manager [EXPERIMENTAL]
%% --------------------------------------------

%% declare_gcp_secrets_config(+Name, +Config)
%  Configure GCP Secret Manager connection.
%
%  Config options:
%    - project(P)           : GCP project ID
%    - impersonate_sa(SA)   : Service account to impersonate
%
declare_gcp_secrets_config(Name, Config) :-
    retractall(gcp_secrets_config_db(Name, _)),
    assertz(gcp_secrets_config_db(Name, Config)).

%% gcp_secrets_config(?Name, ?Config)
%  Query GCP Secret Manager configuration.
%
gcp_secrets_config(Name, Config) :-
    gcp_secrets_config_db(Name, Config).

%% generate_gcp_secret_read(+Source, +SecretId, +Options, -Command)
%  Generate command to read secret from GCP Secret Manager.
%
%  Options:
%    - version(V)           : Specific version (default: latest)
%
generate_gcp_secret_read(Source, SecretId, Options, Command) :-
    gcp_secrets_config(Source, Config),
    option_or_default(project, Config, '', Project),
    option_or_default(impersonate_sa, Config, none, ImpersonateSA),
    option_or_default(version, Options, 'latest', Version),

    % Build impersonation flag
    (ImpersonateSA \= none
    ->  format(atom(ImpersonateFlag), '--impersonate-service-account=~w ', [ImpersonateSA])
    ;   ImpersonateFlag = ''
    ),

    format(atom(Command),
'gcloud secrets versions access ~w --secret=~w --project=~w ~w',
           [Version, SecretId, Project, ImpersonateFlag]).

%% --------------------------------------------
%% Service Secret Bindings [EXPERIMENTAL]
%% --------------------------------------------

%% declare_service_secrets(+Service, +Secrets)
%  Bind secrets to a service for injection.
%
%  Secrets is a list of:
%    - secret(EnvVar, Source, Path)       : Map secret to env var
%    - secret(EnvVar, Source, Path, Field): Map specific field to env var
%    - file_secret(Path, Source, SecretPath): Mount as file
%
declare_service_secrets(Service, Secrets) :-
    retractall(service_secrets_db(Service, _)),
    assertz(service_secrets_db(Service, Secrets)).

%% service_secrets(?Service, ?Secrets)
%  Query service secret bindings.
%
service_secrets(Service, Secrets) :-
    service_secrets_db(Service, Secrets).

%% --------------------------------------------
%% Environment Variable Injection [EXPERIMENTAL]
%% --------------------------------------------

%% generate_secret_env_script(+Service, +Options, -Script)
%  Generate shell script to export secrets as environment variables.
%
generate_secret_env_script(Service, _Options, Script) :-
    service_secrets(Service, Secrets),
    generate_env_exports(Secrets, ExportLines),
    atom_string(Service, ServiceStr),

    format(atom(Script),
'#!/bin/bash
# Secret injection script for ~w
# Generated by UnifyWeaver - Phase 7b Secrets Management
# WARNING: This script is for development/testing only.
# In production, use native secret injection (K8s secrets, Vault Agent, etc.)

set -e

~w

echo "Secrets loaded for ~w"
', [ServiceStr, ExportLines, ServiceStr]).

generate_env_exports([], '').
generate_env_exports([Secret|Rest], Lines) :-
    generate_env_export(Secret, Line),
    generate_env_exports(Rest, RestLines),
    format(atom(Lines), '~w\n~w', [Line, RestLines]).

generate_env_export(secret(EnvVar, Source, Path), Line) :-
    (   vault_config(Source, _)
    ->  generate_vault_read(Source, Path, [format(raw)], Cmd),
        format(atom(Line), 'export ~w=$( ~w )', [EnvVar, Cmd])
    ;   aws_secrets_config(Source, _)
    ->  generate_aws_secret_read(Source, Path, [], Cmd),
        format(atom(Line), 'export ~w=$( ~w )', [EnvVar, Cmd])
    ;   azure_keyvault_config(Source, _)
    ->  generate_azure_secret_read(Source, Path, [], Cmd),
        format(atom(Line), 'export ~w=$( ~w )', [EnvVar, Cmd])
    ;   gcp_secrets_config(Source, _)
    ->  generate_gcp_secret_read(Source, Path, [], Cmd),
        format(atom(Line), 'export ~w=$( ~w )', [EnvVar, Cmd])
    ;   format(atom(Line), '# Unknown source: ~w for ~w', [Source, EnvVar])
    ).

generate_env_export(secret(EnvVar, Source, Path, Field), Line) :-
    (   vault_config(Source, _)
    ->  generate_vault_read(Source, Path, [field(Field)], Cmd),
        format(atom(Line), 'export ~w=$( ~w )', [EnvVar, Cmd])
    ;   aws_secrets_config(Source, _)
    ->  generate_aws_secret_read(Source, Path, [key(Field)], Cmd),
        format(atom(Line), 'export ~w=$( ~w )', [EnvVar, Cmd])
    ;   format(atom(Line), '# Field extraction not supported for ~w', [Source])
    ).

generate_env_export(file_secret(_, _, _), Line) :-
    Line = '# File secrets handled separately'.

%% generate_k8s_secret(+Service, +Options, -Manifest)
%  Generate Kubernetes Secret manifest (for static secrets).
%
%  Note: This creates a placeholder manifest. In production,
%  secrets should be injected by ExternalSecrets or similar.
%
generate_k8s_secret(Service, Options, Manifest) :-
    k8s_config(Service, Config),
    option_or_default(namespace, Config, 'default', Namespace),
    service_secrets(Service, Secrets),
    atom_string(Service, ServiceStr),

    option_or_default(secret_name, Options, none, SecretNameOpt),
    (SecretNameOpt \= none -> SecretName = SecretNameOpt ; format(atom(SecretName), '~w-secrets', [ServiceStr])),

    % Generate data section (placeholders)
    generate_k8s_secret_data(Secrets, DataSection),

    format(atom(Manifest),
'# Generated by UnifyWeaver - Phase 7b Secrets Management
# WARNING: This is a placeholder manifest. Do NOT commit real secrets!
# Use ExternalSecrets or sealed-secrets in production.
apiVersion: v1
kind: Secret
metadata:
  name: ~w
  namespace: ~w
type: Opaque
stringData:
~w', [SecretName, Namespace, DataSection]).

generate_k8s_secret_data([], '  # No secrets configured\n').
generate_k8s_secret_data(Secrets, Section) :-
    Secrets \= [],
    maplist(generate_k8s_secret_item, Secrets, Lines),
    atomic_list_concat(Lines, '\n', Section).

generate_k8s_secret_item(secret(EnvVar, _, _), Line) :-
    format(atom(Line), '  ~w: "PLACEHOLDER_VALUE"', [EnvVar]).
generate_k8s_secret_item(secret(EnvVar, _, _, _), Line) :-
    format(atom(Line), '  ~w: "PLACEHOLDER_VALUE"', [EnvVar]).
generate_k8s_secret_item(file_secret(_, _, _), Line) :-
    Line = '  # file_secret: handled separately'.

%% generate_k8s_external_secret(+Service, +Options, -Manifest)
%  Generate ExternalSecret manifest for external-secrets operator.
%
generate_k8s_external_secret(Service, Options, Manifest) :-
    k8s_config(Service, Config),
    service_secrets(Service, Secrets),
    option_or_default(namespace, Config, 'default', Namespace),
    option_or_default(secret_store, Options, 'vault-backend', SecretStore),
    option_or_default(refresh_interval, Options, '1h', RefreshInterval),
    atom_string(Service, ServiceStr),

    format(atom(SecretName), '~w-secrets', [ServiceStr]),

    % Generate data mappings
    generate_external_secret_data(Secrets, DataSection),

    format(atom(Manifest),
'# Generated by UnifyWeaver - Phase 7b Secrets Management
# ExternalSecret for external-secrets operator
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: ~w
  namespace: ~w
spec:
  refreshInterval: ~w
  secretStoreRef:
    name: ~w
    kind: SecretStore
  target:
    name: ~w
    creationPolicy: Owner
  data:
~w', [SecretName, Namespace, RefreshInterval, SecretStore, SecretName, DataSection]).

generate_external_secret_data([], '  # No secrets configured\n').
generate_external_secret_data(Secrets, Section) :-
    Secrets \= [],
    maplist(generate_external_secret_item, Secrets, Lines),
    atomic_list_concat(Lines, '\n', Section).

generate_external_secret_item(secret(EnvVar, _Source, Path), Line) :-
    format(atom(Line),
'  - secretKey: ~w
    remoteRef:
      key: ~w', [EnvVar, Path]).
generate_external_secret_item(secret(EnvVar, _Source, Path, Field), Line) :-
    format(atom(Line),
'  - secretKey: ~w
    remoteRef:
      key: ~w
      property: ~w', [EnvVar, Path, Field]).
generate_external_secret_item(file_secret(_, _, _), Line) :-
    Line = '  # file_secret: not supported in ExternalSecret'.

%% --------------------------------------------
%% Secret Rotation [EXPERIMENTAL]
%% --------------------------------------------

%% declare_secret_rotation(+Service, +Secret, +Config)
%  Configure secret rotation for a service.
%
%  Config options:
%    - interval(Duration)   : Rotation interval (e.g., '30d', '90d')
%    - notify(Channels)     : Notification channels
%    - pre_rotate(Hook)     : Pre-rotation hook
%    - post_rotate(Hook)    : Post-rotation hook
%
declare_secret_rotation(Service, Secret, Config) :-
    retractall(secret_rotation_db(Service, Secret, _)),
    assertz(secret_rotation_db(Service, Secret, Config)).

%% secret_rotation_config(?Service, ?Secret, ?Config)
%  Query secret rotation configuration.
%
secret_rotation_config(Service, Secret, Config) :-
    secret_rotation_db(Service, Secret, Config).

%% --------------------------------------------
%% Unified Secret Access [EXPERIMENTAL]
%% --------------------------------------------

%% resolve_secret(+Source, +Path, +Options, -Result)
%  Generate command to resolve a secret from any configured source.
%
resolve_secret(Source, Path, Options, Result) :-
    (   vault_config(Source, _)
    ->  generate_vault_read(Source, Path, Options, Cmd),
        Result = vault_secret(Cmd)
    ;   aws_secrets_config(Source, _)
    ->  generate_aws_secret_read(Source, Path, Options, Cmd),
        Result = aws_secret(Cmd)
    ;   azure_keyvault_config(Source, _)
    ->  generate_azure_secret_read(Source, Path, Options, Cmd),
        Result = azure_secret(Cmd)
    ;   gcp_secrets_config(Source, _)
    ->  generate_gcp_secret_read(Source, Path, Options, Cmd),
        Result = gcp_secret(Cmd)
    ;   Result = error(unknown_source(Source))
    ).

%% list_secrets(+Source, +Options, -Secrets)
%  Generate command to list secrets from a source.
%
list_secrets(Source, Options, Secrets) :-
    (   vault_config(Source, Config)
    ->  option_or_default(mount_path, Config, 'secret', MountPath),
        option_or_default(path, Options, '', SubPath),
        (SubPath \= '' -> format(atom(FullPath), '~w/metadata/~w', [MountPath, SubPath])
        ;   format(atom(FullPath), '~w/metadata', [MountPath])),
        format(atom(Cmd), 'vault kv list -format=json ~w', [FullPath]),
        Secrets = vault_list(Cmd)
    ;   aws_secrets_config(Source, Config)
    ->  option_or_default(region, Config, 'us-east-1', Region),
        format(atom(Cmd), 'aws secretsmanager list-secrets --region ~w --query "SecretList[].Name" --output json', [Region]),
        Secrets = aws_list(Cmd)
    ;   azure_keyvault_config(Source, Config)
    ->  option_or_default(vault_name, Config, '', VaultName),
        format(atom(Cmd), 'az keyvault secret list --vault-name ~w --query "[].name" -o json', [VaultName]),
        Secrets = azure_list(Cmd)
    ;   gcp_secrets_config(Source, Config)
    ->  option_or_default(project, Config, '', Project),
        format(atom(Cmd), 'gcloud secrets list --project=~w --format=json', [Project]),
        Secrets = gcp_list(Cmd)
    ;   Secrets = error(unknown_source(Source))
    ).

%% ============================================
%% Phase 7c: Multi-Region Deployment [EXPERIMENTAL]
%% ============================================

%% --------------------------------------------
%% Region Configuration
%% --------------------------------------------

%% declare_region(+Name, +Config)
%  Declare a region configuration.
%  Options:
%    - provider(Provider)      : Cloud provider (aws, gcp, azure)
%    - region_id(Id)          : Provider-specific region ID (e.g., 'us-east-1')
%    - endpoint(URL)          : Custom endpoint URL
%    - availability_zones(List) : Available AZs in this region
%    - latency_zone(Zone)     : Latency zone for geo-routing (e.g., 'NA', 'EU', 'APAC')
%
declare_region(Name, Config) :-
    retractall(region_config_db(Name, _)),
    assertz(region_config_db(Name, Config)).

%% region_config(?Name, ?Config)
%  Query region configuration.
%
region_config(Name, Config) :-
    region_config_db(Name, Config).

%% declare_service_regions(+Service, +RegionConfig)
%  Declare regions for a service.
%  Options:
%    - primary(Region)         : Primary region
%    - secondary(Regions)      : Secondary/failover regions
%    - active_active(bool)     : Whether all regions are active
%    - replication(Config)     : Data replication configuration
%
declare_service_regions(Service, RegionConfig) :-
    retractall(service_regions_db(Service, _)),
    assertz(service_regions_db(Service, RegionConfig)).

%% service_regions(?Service, ?RegionConfig)
%  Query service region configuration.
%
service_regions(Service, RegionConfig) :-
    service_regions_db(Service, RegionConfig).

%% --------------------------------------------
%% Geographic Failover
%% --------------------------------------------

%% declare_failover_policy(+Service, +Policy)
%  Declare failover policy for a service.
%  Options:
%    - strategy(Strategy)      : 'priority' | 'latency' | 'weighted' | 'geolocation'
%    - health_check(Config)    : Health check configuration
%    - failover_threshold(N)   : Number of failures before failover
%    - recovery_threshold(N)   : Number of successes before recovery
%    - dns_ttl(Seconds)        : DNS TTL for failover records
%
declare_failover_policy(Service, Policy) :-
    retractall(failover_policy_db(Service, _)),
    assertz(failover_policy_db(Service, Policy)).

%% failover_policy(?Service, ?Policy)
%  Query failover policy.
%
failover_policy(Service, Policy) :-
    failover_policy_db(Service, Policy).

%% select_region(+Service, +Options, -Region)
%  Select the best region for a service based on policy.
%  Options:
%    - source_location(Loc)    : Source location for latency-based selection
%    - prefer_healthy(bool)    : Only select healthy regions
%
select_region(Service, Options, Region) :-
    service_regions(Service, RegionConfig),
    failover_policy(Service, Policy),
    option_or_default(strategy, Policy, priority, Strategy),
    select_region_by_strategy(Service, RegionConfig, Strategy, Options, Region).

%% select_region_by_strategy(+Service, +RegionConfig, +Strategy, +Options, -Region)
%  Select region based on strategy.
%
select_region_by_strategy(_Service, RegionConfig, priority, Options, Region) :-
    option_or_default(primary, RegionConfig, none, Primary),
    option_or_default(secondary, RegionConfig, [], Secondary),
    option_or_default(prefer_healthy, Options, true, PreferHealthy),
    (   PreferHealthy == true
    ->  % Try primary first if healthy
        (   region_is_healthy(Primary)
        ->  Region = Primary
        ;   % Try secondary regions in order
            member(Region, Secondary),
            region_is_healthy(Region)
        )
    ;   Region = Primary
    ).

select_region_by_strategy(_Service, RegionConfig, latency, Options, Region) :-
    option_or_default(primary, RegionConfig, none, Primary),
    option_or_default(secondary, RegionConfig, [], Secondary),
    option_or_default(source_location, Options, unknown, SourceLoc),
    AllRegions = [Primary|Secondary],
    (   SourceLoc \= unknown
    ->  % Select by latency zone
        select_by_latency_zone(AllRegions, SourceLoc, Region)
    ;   Region = Primary
    ).

select_region_by_strategy(_Service, RegionConfig, weighted, _Options, Region) :-
    option_or_default(primary, RegionConfig, none, Primary),
    % For weighted, just return primary (would need weights in real impl)
    Region = Primary.

select_region_by_strategy(_Service, RegionConfig, geolocation, Options, Region) :-
    option_or_default(source_location, Options, unknown, SourceLoc),
    option_or_default(geo_mappings, RegionConfig, [], Mappings),
    (   member(SourceLoc-MappedRegion, Mappings)
    ->  Region = MappedRegion
    ;   option_or_default(primary, RegionConfig, none, Region)
    ).

%% region_is_healthy(+Region)
%  Check if a region is healthy (based on stored status).
%
region_is_healthy(Region) :-
    (   region_status_db(_, Region, healthy, _)
    ->  true
    ;   % Default to healthy if no status recorded
        true
    ).

%% select_by_latency_zone(+Regions, +SourceLoc, -Region)
%  Select region by latency zone matching.
%
select_by_latency_zone(Regions, SourceLoc, Region) :-
    member(Region, Regions),
    region_config(Region, Config),
    option_or_default(latency_zone, Config, unknown, Zone),
    Zone = SourceLoc,
    !.
select_by_latency_zone([Region|_], _, Region).

%% failover_to_region(+Service, +Region, -Result)
%  Initiate failover to a specific region.
%  Generates the commands needed for failover.
%
failover_to_region(Service, Region, Result) :-
    service_regions(Service, RegionConfig),
    option_or_default(primary, RegionConfig, none, CurrentPrimary),
    (   CurrentPrimary == Region
    ->  Result = already_primary(Region)
    ;   % Generate failover commands
        generate_failover_commands(Service, CurrentPrimary, Region, Commands),
        Result = failover_commands(Commands)
    ).

%% generate_failover_commands(+Service, +FromRegion, +ToRegion, -Commands)
%  Generate commands for failover.
%
generate_failover_commands(Service, FromRegion, ToRegion, Commands) :-
    traffic_policy(Service, Policy),
    option_or_default(dns_provider, Policy, route53, DnsProvider),
    (   DnsProvider == route53
    ->  format(atom(Cmd1), '# Failover ~w from ~w to ~w', [Service, FromRegion, ToRegion]),
        format(atom(Cmd2), 'aws route53 change-resource-record-sets --hosted-zone-id $ZONE_ID --change-batch file://failover-~w.json', [Service]),
        Commands = [Cmd1, Cmd2]
    ;   DnsProvider == cloudflare
    ->  format(atom(Cmd1), '# Failover ~w from ~w to ~w', [Service, FromRegion, ToRegion]),
        format(atom(Cmd2), 'curl -X PATCH "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records/$RECORD_ID" -H "Authorization: Bearer $CF_TOKEN" -H "Content-Type: application/json" --data \'{"content":"~w"}\'', [ToRegion]),
        Commands = [Cmd1, Cmd2]
    ;   Commands = ['# Unknown DNS provider']
    ).

%% --------------------------------------------
%% Multi-Region Deployment
%% --------------------------------------------

%% deploy_to_region(+Service, +Region, +Options, -Result)
%  Generate deployment commands for a specific region.
%
deploy_to_region(Service, Region, Options, Result) :-
    region_config(Region, RegionCfg),
    option_or_default(provider, RegionCfg, aws, Provider),
    option_or_default(region_id, RegionCfg, 'us-east-1', RegionId),
    generate_region_deploy_commands(Service, Provider, RegionId, Options, Commands),
    Result = deploy_commands(Region, Commands).

%% generate_region_deploy_commands(+Service, +Provider, +RegionId, +Options, -Commands)
%  Generate provider-specific deployment commands.
%
generate_region_deploy_commands(Service, aws, RegionId, _Options, Commands) :-
    format(atom(Cmd1), 'export AWS_REGION=~w', [RegionId]),
    format(atom(Cmd2), 'aws ecs update-service --cluster ~w-cluster --service ~w --force-new-deployment --region ~w', [Service, Service, RegionId]),
    Commands = [Cmd1, Cmd2].

generate_region_deploy_commands(Service, gcp, RegionId, _Options, Commands) :-
    format(atom(Cmd1), 'gcloud config set compute/region ~w', [RegionId]),
    format(atom(Cmd2), 'gcloud run deploy ~w --region ~w --image gcr.io/$PROJECT_ID/~w:latest', [Service, RegionId, Service]),
    Commands = [Cmd1, Cmd2].

generate_region_deploy_commands(Service, azure, RegionId, _Options, Commands) :-
    format(atom(Cmd1), 'az configure --defaults location=~w', [RegionId]),
    format(atom(Cmd2), 'az container create --resource-group ~w-rg --name ~w --location ~w', [Service, Service, RegionId]),
    Commands = [Cmd1, Cmd2].

%% deploy_to_all_regions(+Service, +Options, -Results)
%  Generate deployment commands for all configured regions.
%
deploy_to_all_regions(Service, Options, Results) :-
    service_regions(Service, RegionConfig),
    option_or_default(primary, RegionConfig, none, Primary),
    option_or_default(secondary, RegionConfig, [], Secondary),
    AllRegions = [Primary|Secondary],
    findall(
        Result,
        (   member(Region, AllRegions),
            Region \= none,
            deploy_to_region(Service, Region, Options, Result)
        ),
        Results
    ).

%% region_status(+Service, +Region, -Status)
%  Get the current status of a service in a region.
%  Returns command to check status.
%
region_status(Service, Region, Status) :-
    region_config(Region, RegionCfg),
    option_or_default(provider, RegionCfg, aws, Provider),
    option_or_default(region_id, RegionCfg, 'us-east-1', RegionId),
    generate_status_command(Service, Provider, RegionId, StatusCmd),
    Status = status_command(Region, StatusCmd).

%% generate_status_command(+Service, +Provider, +RegionId, -Command)
%  Generate provider-specific status check command.
%
generate_status_command(Service, aws, RegionId, Command) :-
    format(atom(Command), 'aws ecs describe-services --cluster ~w-cluster --services ~w --region ~w --query "services[0].status"', [Service, Service, RegionId]).

generate_status_command(Service, gcp, RegionId, Command) :-
    format(atom(Command), 'gcloud run services describe ~w --region ~w --format="value(status.conditions[0].status)"', [Service, RegionId]).

generate_status_command(Service, azure, RegionId, Command) :-
    format(atom(Command), 'az container show --resource-group ~w-rg --name ~w --query "instanceView.state" --location ~w', [Service, Service, RegionId]).

%% generate_region_config(+Service, +Options, -Config)
%  Generate a region configuration file (e.g., for Terraform).
%
generate_region_config(Service, Options, Config) :-
    service_regions(Service, RegionConfig),
    option_or_default(format, Options, terraform, Format),
    (   Format == terraform
    ->  generate_terraform_region_config(Service, RegionConfig, Config)
    ;   Format == json
    ->  generate_json_region_config(Service, RegionConfig, Config)
    ;   Config = 'Unknown format'
    ).

%% generate_terraform_region_config(+Service, +RegionConfig, -Config)
%  Generate Terraform configuration for multi-region.
%
generate_terraform_region_config(Service, RegionConfig, Config) :-
    option_or_default(primary, RegionConfig, 'us-east-1', Primary),
    option_or_default(secondary, RegionConfig, [], Secondary),
    format(atom(Header), '# Multi-region configuration for ~w\n\n', [Service]),
    format(atom(PrimaryBlock), 'module "~w_primary" {\n  source = "./modules/service"\n  region = "~w"\n  is_primary = true\n}\n\n', [Service, Primary]),
    format_secondary_blocks(Service, Secondary, SecondaryBlocks),
    atom_concat(Header, PrimaryBlock, Temp),
    atom_concat(Temp, SecondaryBlocks, Config).

%% format_secondary_blocks(+Service, +Regions, -Blocks)
%  Format Terraform blocks for secondary regions.
%
format_secondary_blocks(_, [], '').
format_secondary_blocks(Service, [Region|Rest], Blocks) :-
    format(atom(Block), 'module "~w_~w" {\n  source = "./modules/service"\n  region = "~w"\n  is_primary = false\n}\n\n', [Service, Region, Region]),
    format_secondary_blocks(Service, Rest, RestBlocks),
    atom_concat(Block, RestBlocks, Blocks).

%% generate_json_region_config(+Service, +RegionConfig, -Config)
%  Generate JSON configuration for multi-region.
%
generate_json_region_config(Service, RegionConfig, Config) :-
    option_or_default(primary, RegionConfig, 'us-east-1', Primary),
    option_or_default(secondary, RegionConfig, [], Secondary),
    format(atom(Config), '{\n  "service": "~w",\n  "primary": "~w",\n  "secondary": ~w\n}', [Service, Primary, Secondary]).

%% --------------------------------------------
%% Traffic Management
%% --------------------------------------------

%% declare_traffic_policy(+Service, +Policy)
%  Declare traffic routing policy.
%  Options:
%    - dns_provider(Provider)  : 'route53' | 'cloudflare' | 'gcp_dns'
%    - routing_policy(Policy)  : 'simple' | 'weighted' | 'latency' | 'geolocation' | 'failover'
%    - weights(List)           : Region weights for weighted routing
%    - health_check_id(Id)     : Associated health check
%
declare_traffic_policy(Service, Policy) :-
    retractall(traffic_policy_db(Service, _)),
    assertz(traffic_policy_db(Service, Policy)).

%% traffic_policy(?Service, ?Policy)
%  Query traffic policy.
%
traffic_policy(Service, Policy) :-
    traffic_policy_db(Service, Policy).

%% generate_route53_config(+Service, +Options, -Config)
%  Generate AWS Route53 configuration.
%
generate_route53_config(Service, Options, Config) :-
    service_regions(Service, RegionConfig),
    traffic_policy(Service, TrafficPolicy),
    option_or_default(routing_policy, TrafficPolicy, failover, RoutingPolicy),
    option_or_default(hosted_zone, Options, '$HOSTED_ZONE_ID', HostedZone),
    option_or_default(domain, Options, 'example.com', Domain),
    option_or_default(primary, RegionConfig, 'us-east-1', Primary),
    option_or_default(secondary, RegionConfig, [], Secondary),
    generate_route53_records(Service, Domain, HostedZone, Primary, Secondary, RoutingPolicy, Config).

%% generate_route53_records(+Service, +Domain, +Zone, +Primary, +Secondary, +Policy, -Config)
%  Generate Route53 record configuration.
%
generate_route53_records(Service, Domain, Zone, Primary, Secondary, failover, Config) :-
    format(atom(Header), '{\n  "Comment": "Multi-region failover for ~w",\n  "Changes": [\n', [Service]),
    format(atom(PrimaryRecord), '    {\n      "Action": "UPSERT",\n      "ResourceRecordSet": {\n        "Name": "~w.~w",\n        "Type": "A",\n        "SetIdentifier": "primary",\n        "Failover": "PRIMARY",\n        "TTL": 60,\n        "ResourceRecords": [{"Value": "~w-endpoint"}],\n        "HealthCheckId": "$HEALTH_CHECK_ID"\n      }\n    }', [Service, Domain, Primary]),
    (   Secondary = [SecondaryRegion|_]
    ->  format(atom(SecondaryRecord), ',\n    {\n      "Action": "UPSERT",\n      "ResourceRecordSet": {\n        "Name": "~w.~w",\n        "Type": "A",\n        "SetIdentifier": "secondary",\n        "Failover": "SECONDARY",\n        "TTL": 60,\n        "ResourceRecords": [{"Value": "~w-endpoint"}]\n      }\n    }', [Service, Domain, SecondaryRegion])
    ;   SecondaryRecord = ''
    ),
    format(atom(Footer), '\n  ]\n}', []),
    atomic_list_concat([Header, PrimaryRecord, SecondaryRecord, Footer], Config),
    _ = Zone. % Suppress unused warning

generate_route53_records(Service, Domain, _Zone, Primary, Secondary, weighted, Config) :-
    format(atom(Header), '{\n  "Comment": "Weighted routing for ~w",\n  "Changes": [\n', [Service]),
    format(atom(PrimaryRecord), '    {\n      "Action": "UPSERT",\n      "ResourceRecordSet": {\n        "Name": "~w.~w",\n        "Type": "A",\n        "SetIdentifier": "~w",\n        "Weight": 70,\n        "TTL": 60,\n        "ResourceRecords": [{"Value": "~w-endpoint"}]\n      }\n    }', [Service, Domain, Primary, Primary]),
    format_weighted_secondary(Service, Domain, Secondary, 30, SecondaryRecords),
    format(atom(Footer), '\n  ]\n}', []),
    atomic_list_concat([Header, PrimaryRecord, SecondaryRecords, Footer], Config).

generate_route53_records(Service, Domain, _Zone, Primary, _Secondary, simple, Config) :-
    format(atom(Config), '{\n  "Comment": "Simple routing for ~w",\n  "Changes": [\n    {\n      "Action": "UPSERT",\n      "ResourceRecordSet": {\n        "Name": "~w.~w",\n        "Type": "A",\n        "TTL": 300,\n        "ResourceRecords": [{"Value": "~w-endpoint"}]\n      }\n    }\n  ]\n}', [Service, Service, Domain, Primary]).

%% format_weighted_secondary(+Service, +Domain, +Regions, +Weight, -Records)
%  Format weighted records for secondary regions.
%
format_weighted_secondary(_, _, [], _, '').
format_weighted_secondary(Service, Domain, [Region|Rest], Weight, Records) :-
    format(atom(Record), ',\n    {\n      "Action": "UPSERT",\n      "ResourceRecordSet": {\n        "Name": "~w.~w",\n        "Type": "A",\n        "SetIdentifier": "~w",\n        "Weight": ~w,\n        "TTL": 60,\n        "ResourceRecords": [{"Value": "~w-endpoint"}]\n      }\n    }', [Service, Domain, Region, Weight, Region]),
    NewWeight is Weight div 2,
    format_weighted_secondary(Service, Domain, Rest, NewWeight, RestRecords),
    atom_concat(Record, RestRecords, Records).

%% generate_cloudflare_config(+Service, +Options, -Config)
%  Generate Cloudflare DNS configuration.
%
generate_cloudflare_config(Service, Options, Config) :-
    service_regions(Service, RegionConfig),
    option_or_default(domain, Options, 'example.com', Domain),
    option_or_default(primary, RegionConfig, 'us-east-1', Primary),
    format(atom(Config), '{\n  "type": "A",\n  "name": "~w.~w",\n  "content": "~w-endpoint",\n  "ttl": 1,\n  "proxied": true\n}', [Service, Domain, Primary]).

%% ============================================
%% Phase 7c: Cloud Functions [EXPERIMENTAL]
%% ============================================

%% --------------------------------------------
%% AWS Lambda
%% --------------------------------------------

%% declare_lambda_config(+Function, +Config)
%  Declare AWS Lambda function configuration.
%  Options:
%    - runtime(Runtime)        : 'python3.11' | 'nodejs20.x' | 'go1.x' | 'dotnet8'
%    - handler(Handler)        : Function handler (e.g., 'index.handler')
%    - memory(MB)              : Memory allocation (128-10240)
%    - timeout(Seconds)        : Function timeout (1-900)
%    - role(ARN)               : IAM execution role ARN
%    - environment(Vars)       : Environment variables
%    - vpc_config(Config)      : VPC configuration
%    - layers(List)            : Lambda layer ARNs
%
declare_lambda_config(Function, Config) :-
    retractall(lambda_config_db(Function, _)),
    assertz(lambda_config_db(Function, Config)).

%% lambda_config(?Function, ?Config)
%  Query Lambda configuration.
%
lambda_config(Function, Config) :-
    lambda_config_db(Function, Config).

%% generate_lambda_function(+Function, +Options, -Package)
%  Generate Lambda function package structure.
%
generate_lambda_function(Function, Options, Package) :-
    lambda_config(Function, Config),
    option_or_default(runtime, Config, 'python3.11', Runtime),
    option_or_default(handler, Config, 'index.handler', Handler),
    generate_lambda_handler(Function, Runtime, Handler, Options, HandlerCode),
    Package = lambda_package(Function, Runtime, HandlerCode).

%% generate_lambda_handler(+Function, +Runtime, +Handler, +Options, -Code)
%  Generate Lambda handler boilerplate.
%
generate_lambda_handler(Function, Runtime, Handler, _Options, Code) :-
    (   sub_atom(Runtime, 0, _, _, python)
    ->  format(atom(Code), '# Lambda function: ~w\n# Handler: ~w\n\nimport json\nimport logging\n\nlogger = logging.getLogger()\nlogger.setLevel(logging.INFO)\n\ndef handler(event, context):\n    """\n    AWS Lambda handler for ~w\n    """\n    logger.info(f"Event: {json.dumps(event)}")\n    \n    # TODO: Implement function logic\n    \n    return {\n        "statusCode": 200,\n        "body": json.dumps({"message": "Success"})\n    }\n', [Function, Handler, Function])
    ;   sub_atom(Runtime, 0, _, _, nodejs)
    ->  format(atom(Code), '// Lambda function: ~w\n// Handler: ~w\n\nexports.handler = async (event, context) => {\n    console.log(\'Event:\', JSON.stringify(event));\n    \n    // TODO: Implement function logic\n    \n    return {\n        statusCode: 200,\n        body: JSON.stringify({ message: \'Success\' })\n    };\n};\n', [Function, Handler])
    ;   sub_atom(Runtime, 0, _, _, go)
    ->  format(atom(Code), '// Lambda function: ~w\npackage main\n\nimport (\n    "context"\n    "encoding/json"\n    "github.com/aws/aws-lambda-go/events"\n    "github.com/aws/aws-lambda-go/lambda"\n)\n\nfunc handler(ctx context.Context, event events.APIGatewayProxyRequest) (events.APIGatewayProxyResponse, error) {\n    // TODO: Implement function logic\n    \n    return events.APIGatewayProxyResponse{\n        StatusCode: 200,\n        Body:       `{"message":"Success"}`,\n    }, nil\n}\n\nfunc main() {\n    lambda.Start(handler)\n}\n', [Function])
    ;   format(atom(Code), '# Unsupported runtime: ~w', [Runtime])
    ).

%% generate_lambda_deploy(+Function, +Options, -Commands)
%  Generate Lambda deployment commands.
%
generate_lambda_deploy(Function, Options, Commands) :-
    lambda_config(Function, Config),
    option_or_default(runtime, Config, 'python3.11', Runtime),
    option_or_default(handler, Config, 'index.handler', Handler),
    option_or_default(memory, Config, 256, MemoryRaw),
    option_or_default(timeout, Config, 30, TimeoutRaw),
    option_or_default(role, Config, '$LAMBDA_ROLE_ARN', Role),
    option_or_default(region, Options, 'us-east-1', Region),
    % Convert integers to atoms for format/3
    term_to_atom(MemoryRaw, Memory),
    term_to_atom(TimeoutRaw, Timeout),
    format(atom(Cmd1), '# Deploy Lambda function: ~w', [Function]),
    format(atom(Cmd2), 'cd ~w && zip -r function.zip .', [Function]),
    format(atom(Cmd3), 'aws lambda create-function --function-name ~w --runtime ~w --handler ~w --role ~w --zip-file fileb://function.zip --memory-size ~w --timeout ~w --region ~w', [Function, Runtime, Handler, Role, Memory, Timeout, Region]),
    format(atom(Cmd4), '# Or update existing function:'),
    format(atom(Cmd5), 'aws lambda update-function-code --function-name ~w --zip-file fileb://function.zip --region ~w', [Function, Region]),
    Commands = [Cmd1, Cmd2, Cmd3, Cmd4, Cmd5].

%% generate_sam_template(+Function, +Options, -Template)
%  Generate AWS SAM template for Lambda function.
%
generate_sam_template(Function, Options, Template) :-
    lambda_config(Function, Config),
    option_or_default(runtime, Config, 'python3.11', Runtime),
    option_or_default(handler, Config, 'index.handler', Handler),
    option_or_default(memory, Config, 256, Memory),
    option_or_default(timeout, Config, 30, Timeout),
    option_or_default(description, Options, 'Lambda function', Description),
    format(atom(Template), 'AWSTemplateFormatVersion: \'2010-09-09\'\nTransform: AWS::Serverless-2016-10-31\nDescription: ~w\n\nGlobals:\n  Function:\n    Timeout: ~w\n    MemorySize: ~w\n\nResources:\n  ~wFunction:\n    Type: AWS::Serverless::Function\n    Properties:\n      FunctionName: ~w\n      Runtime: ~w\n      Handler: ~w\n      CodeUri: ./\n      Description: ~w\n      Events:\n        Api:\n          Type: Api\n          Properties:\n            Path: /~w\n            Method: ANY\n\nOutputs:\n  ~wApi:\n    Description: API Gateway endpoint URL\n    Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/~w/"\n  ~wFunction:\n    Description: Lambda Function ARN\n    Value: !GetAtt ~wFunction.Arn\n', [Description, Timeout, Memory, Function, Function, Runtime, Handler, Description, Function, Function, Function, Function, Function]).

%% --------------------------------------------
%% Google Cloud Functions
%% --------------------------------------------

%% declare_gcf_config(+Function, +Config)
%  Declare Google Cloud Functions configuration.
%  Options:
%    - runtime(Runtime)        : 'python311' | 'nodejs20' | 'go121' | 'dotnet8'
%    - entry_point(Name)       : Function entry point
%    - memory(MB)              : Memory allocation (128-8192)
%    - timeout(Seconds)        : Function timeout (1-540)
%    - region(Region)          : Deployment region
%    - trigger(Type)           : 'http' | 'pubsub' | 'storage' | 'firestore'
%    - environment(Vars)       : Environment variables
%
declare_gcf_config(Function, Config) :-
    retractall(gcf_config_db(Function, _)),
    assertz(gcf_config_db(Function, Config)).

%% gcf_config(?Function, ?Config)
%  Query GCF configuration.
%
gcf_config(Function, Config) :-
    gcf_config_db(Function, Config).

%% generate_gcf_deploy(+Function, +Options, -Commands)
%  Generate Google Cloud Functions deployment commands.
%
generate_gcf_deploy(Function, Options, Commands) :-
    gcf_config(Function, Config),
    option_or_default(runtime, Config, 'python311', Runtime),
    option_or_default(entry_point, Config, 'main', EntryPoint),
    option_or_default(memory, Config, 256, Memory),
    option_or_default(timeout, Config, 60, Timeout),
    option_or_default(region, Config, 'us-central1', Region),
    option_or_default(trigger, Config, http, Trigger),
    option_or_default(project, Options, '$GCP_PROJECT', Project),
    format(atom(Cmd1), '# Deploy Google Cloud Function: ~w', [Function]),
    (   Trigger == http
    ->  format(atom(TriggerOpt), '--trigger-http --allow-unauthenticated', [])
    ;   Trigger == pubsub
    ->  format(atom(TriggerOpt), '--trigger-topic ~w-topic', [Function])
    ;   format(atom(TriggerOpt), '--trigger-http', [])
    ),
    format(atom(Cmd2), 'gcloud functions deploy ~w --runtime ~w --entry-point ~w --memory ~wMB --timeout ~ws --region ~w ~w --project ~w', [Function, Runtime, EntryPoint, Memory, Timeout, Region, TriggerOpt, Project]),
    Commands = [Cmd1, Cmd2].

%% --------------------------------------------
%% Azure Functions
%% --------------------------------------------

%% declare_azure_func_config(+Function, +Config)
%  Declare Azure Functions configuration.
%  Options:
%    - runtime(Runtime)        : 'python' | 'node' | 'dotnet' | 'java'
%    - version(Version)        : Runtime version
%    - os(OS)                  : 'windows' | 'linux'
%    - plan(Plan)              : 'consumption' | 'premium' | 'dedicated'
%    - resource_group(Name)    : Azure resource group
%    - storage_account(Name)   : Storage account name
%
declare_azure_func_config(Function, Config) :-
    retractall(azure_func_config_db(Function, _)),
    assertz(azure_func_config_db(Function, Config)).

%% azure_func_config(?Function, ?Config)
%  Query Azure Functions configuration.
%
azure_func_config(Function, Config) :-
    azure_func_config_db(Function, Config).

%% generate_azure_func_deploy(+Function, +Options, -Commands)
%  Generate Azure Functions deployment commands.
%
generate_azure_func_deploy(Function, Options, Commands) :-
    azure_func_config(Function, Config),
    option_or_default(runtime, Config, 'python', Runtime),
    option_or_default(version, Config, '3.11', Version),
    option_or_default(os, Config, 'linux', OS),
    option_or_default(resource_group, Config, '$AZURE_RG', ResourceGroup),
    option_or_default(storage_account, Config, '$AZURE_STORAGE', Storage),
    option_or_default(location, Options, 'eastus', Location),
    format(atom(Cmd1), '# Deploy Azure Function: ~w', [Function]),
    format(atom(Cmd2), 'az functionapp create --name ~w --resource-group ~w --storage-account ~w --runtime ~w --runtime-version ~w --os-type ~w --consumption-plan-location ~w', [Function, ResourceGroup, Storage, Runtime, Version, OS, Location]),
    format(atom(Cmd3), 'func azure functionapp publish ~w', [Function]),
    Commands = [Cmd1, Cmd2, Cmd3].

%% --------------------------------------------
%% API Gateway Integration
%% --------------------------------------------

%% declare_api_gateway(+Name, +Config)
%  Declare API Gateway configuration.
%  Options:
%    - provider(Provider)      : 'aws' | 'gcp' | 'azure'
%    - name(Name)              : Gateway name
%    - description(Desc)       : Description
%    - endpoints(List)         : List of endpoint configurations
%    - auth(Config)            : Authentication configuration
%    - cors(Config)            : CORS configuration
%
declare_api_gateway(Name, Config) :-
    retractall(api_gateway_config_db(Name, _)),
    assertz(api_gateway_config_db(Name, Config)).

%% api_gateway_config(?Name, ?Config)
%  Query API Gateway configuration.
%
api_gateway_config(Name, Config) :-
    api_gateway_config_db(Name, Config).

%% generate_api_gateway_config(+Name, +Options, -Config)
%  Generate API Gateway configuration.
%
generate_api_gateway_config(Name, Options, Config) :-
    api_gateway_config(Name, GatewayConfig),
    option_or_default(provider, GatewayConfig, aws, Provider),
    (   Provider == aws
    ->  generate_aws_api_gateway(Name, GatewayConfig, Options, Config)
    ;   Provider == gcp
    ->  generate_gcp_api_gateway(Name, GatewayConfig, Options, Config)
    ;   Provider == azure
    ->  generate_azure_api_gateway(Name, GatewayConfig, Options, Config)
    ;   Config = 'Unknown provider'
    ).

%% generate_aws_api_gateway(+Name, +GatewayConfig, +Options, -Config)
%  Generate AWS API Gateway configuration.
%
generate_aws_api_gateway(Name, GatewayConfig, _Options, Config) :-
    option_or_default(description, GatewayConfig, 'API Gateway', Description),
    option_or_default(endpoints, GatewayConfig, [], Endpoints),
    format_aws_endpoints(Endpoints, EndpointConfigs),
    format(atom(Config), '{\n  "swagger": "2.0",\n  "info": {\n    "title": "~w",\n    "description": "~w",\n    "version": "1.0"\n  },\n  "basePath": "/v1",\n  "schemes": ["https"],\n  "paths": {\n~w  }\n}', [Name, Description, EndpointConfigs]).

%% format_aws_endpoints(+Endpoints, -Config)
%  Format AWS API Gateway endpoints.
%
format_aws_endpoints([], '').
format_aws_endpoints([endpoint(Path, Method, Function)|Rest], Config) :-
    format(atom(EndpointConfig), '    "~w": {\n      "~w": {\n        "x-amazon-apigateway-integration": {\n          "uri": "arn:aws:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/arn:aws:lambda:${AWS::Region}:${AWS::AccountId}:function:~w/invocations",\n          "type": "aws_proxy",\n          "httpMethod": "POST"\n        }\n      }\n    }', [Path, Method, Function]),
    format_aws_endpoints(Rest, RestConfig),
    (   RestConfig \= ''
    ->  format(atom(Config), '~w,\n~w', [EndpointConfig, RestConfig])
    ;   Config = EndpointConfig
    ).

%% generate_gcp_api_gateway(+Name, +GatewayConfig, +Options, -Config)
%  Generate GCP API Gateway configuration.
%
generate_gcp_api_gateway(Name, GatewayConfig, _Options, Config) :-
    option_or_default(description, GatewayConfig, 'API Gateway', Description),
    format(atom(Config), 'swagger: "2.0"\ninfo:\n  title: ~w\n  description: ~w\n  version: "1.0.0"\nhost: "~w.apigateway.$PROJECT_ID.cloud.goog"\nschemes:\n  - https\npaths:\n  /hello:\n    get:\n      summary: Hello endpoint\n      operationId: hello\n      x-google-backend:\n        address: https://$REGION-$PROJECT_ID.cloudfunctions.net/~w\n      responses:\n        "200":\n          description: Success\n', [Name, Description, Name, Name]).

%% generate_azure_api_gateway(+Name, +GatewayConfig, +Options, -Config)
%  Generate Azure API Management configuration.
%
generate_azure_api_gateway(Name, GatewayConfig, _Options, Config) :-
    option_or_default(description, GatewayConfig, 'API Gateway', Description),
    format(atom(Config), '<?xml version="1.0" encoding="UTF-8"?>\n<api-management>\n  <api name="~w">\n    <description>~w</description>\n    <subscription-required>false</subscription-required>\n    <operations>\n      <operation name="get-hello" method="GET" url-template="/hello">\n        <backend-service>\n          <url>https://~w.azurewebsites.net/api/hello</url>\n        </backend-service>\n      </operation>\n    </operations>\n  </api>\n</api-management>\n', [Name, Description, Name]).

%% generate_openapi_spec(+Gateway, +Options, -Spec)
%  Generate OpenAPI specification for an API Gateway.
%
generate_openapi_spec(Gateway, Options, Spec) :-
    api_gateway_config(Gateway, GatewayConfig),
    option_or_default(version, Options, '3.0.0', Version),
    option_or_default(description, GatewayConfig, 'API Gateway', Description),
    option_or_default(endpoints, GatewayConfig, [], Endpoints),
    format_openapi_paths(Endpoints, Paths),
    format(atom(Spec), 'openapi: "~w"\ninfo:\n  title: ~w\n  description: ~w\n  version: "1.0.0"\nservers:\n  - url: https://api.example.com/v1\npaths:\n~w', [Version, Gateway, Description, Paths]).

%% format_openapi_paths(+Endpoints, -Paths)
%  Format OpenAPI paths.
%
format_openapi_paths([], '').
format_openapi_paths([endpoint(Path, Method, _Function)|Rest], Paths) :-
    format(atom(PathConfig), '  ~w:\n    ~w:\n      summary: ~w endpoint\n      responses:\n        "200":\n          description: Successful response\n', [Path, Method, Path]),
    format_openapi_paths(Rest, RestPaths),
    atom_concat(PathConfig, RestPaths, Paths).

%% --------------------------------------------
%% Unified Serverless Deployment
%% --------------------------------------------

%% deploy_function(+Function, +Options, -Result)
%  Deploy a serverless function to any configured provider.
%
deploy_function(Function, Options, Result) :-
    (   lambda_config(Function, _)
    ->  generate_lambda_deploy(Function, Options, Commands),
        Result = lambda_deploy(Commands)
    ;   gcf_config(Function, _)
    ->  generate_gcf_deploy(Function, Options, Commands),
        Result = gcf_deploy(Commands)
    ;   azure_func_config(Function, _)
    ->  generate_azure_func_deploy(Function, Options, Commands),
        Result = azure_deploy(Commands)
    ;   Result = error(unknown_function(Function))
    ).

%% invoke_function(+Function, +Payload, +Options, -Result)
%  Generate command to invoke a serverless function.
%
invoke_function(Function, Payload, Options, Result) :-
    (   lambda_config(Function, Config)
    ->  option_or_default(region, Options, 'us-east-1', Region),
        format(atom(Cmd), 'aws lambda invoke --function-name ~w --payload \'~w\' --region ~w /dev/stdout', [Function, Payload, Region]),
        Result = invoke_command(aws, Cmd),
        _ = Config  % Suppress unused warning
    ;   gcf_config(Function, Config)
    ->  option_or_default(region, Config, 'us-central1', Region),
        option_or_default(project, Options, '$GCP_PROJECT', Project),
        format(atom(Cmd), 'gcloud functions call ~w --data \'~w\' --region ~w --project ~w', [Function, Payload, Region, Project]),
        Result = invoke_command(gcp, Cmd)
    ;   azure_func_config(Function, _)
    ->  format(atom(Cmd), 'curl -X POST "https://~w.azurewebsites.net/api/~w" -H "Content-Type: application/json" -d \'~w\'', [Function, Function, Payload]),
        Result = invoke_command(azure, Cmd)
    ;   Result = error(unknown_function(Function))
    ).

%% function_logs(+Function, +Options, -Logs)
%  Generate command to retrieve function logs.
%
function_logs(Function, Options, Logs) :-
    (   lambda_config(Function, _)
    ->  option_or_default(region, Options, 'us-east-1', Region),
        option_or_default(tail, Options, 100, Tail),
        format(atom(Cmd), 'aws logs tail /aws/lambda/~w --follow --since 1h --region ~w | head -~w', [Function, Region, Tail]),
        Logs = log_command(aws, Cmd)
    ;   gcf_config(Function, Config)
    ->  option_or_default(region, Config, 'us-central1', Region),
        option_or_default(limit, Options, 100, Limit),
        format(atom(Cmd), 'gcloud functions logs read ~w --region ~w --limit ~w', [Function, Region, Limit]),
        Logs = log_command(gcp, Cmd)
    ;   azure_func_config(Function, Config)
    ->  option_or_default(resource_group, Config, '$AZURE_RG', ResourceGroup),
        format(atom(Cmd), 'az webapp log tail --name ~w --resource-group ~w', [Function, ResourceGroup]),
        Logs = log_command(azure, Cmd)
    ;   Logs = error(unknown_function(Function))
    ).
