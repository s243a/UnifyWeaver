# Cross-Target Glue Phase 6: Advanced Features Design

## Overview

Phase 6 extends the cross-target glue system with production-ready features:

1. **Automatic Deployment** - Deploy and manage remote services via SSH
2. **Lifecycle Management** - Service startup, shutdown, and redeployment
3. **Error Handling** - Retry, fallback, and timeout mechanisms
4. **Monitoring** - Metrics collection and health checks

## Part 1: Automatic Deployment

### Design Goals

1. **Security by Default** - Remote services require encryption (SSH, HTTPS, VPN)
2. **Agent-Based Deployment** - Use SSH agent for passwordless authentication
3. **Lifecycle Awareness** - Track service state, handle graceful shutdown
4. **Change Detection** - Redeploy when source code changes
5. **Minimal Configuration** - Sensible defaults with override capability

### Deployment Methods

#### 1.1 SSH Deployment (Primary)

Uses SSH agent forwarding for secure, passwordless deployment.

```prolog
:- deploy_method(ssh, [
    host('worker.example.com'),
    user('deploy'),
    agent(true),                    % Use SSH agent (default)
    key_file('~/.ssh/deploy_key'),  % Or explicit key
    remote_dir('/opt/unifyweaver/services')
]).
```

**Requirements:**
- SSH agent running with key loaded (`ssh-add`)
- Remote host in `known_hosts`
- User has write access to deployment directory

#### 1.2 Container Deployment (Future)

```prolog
:- deploy_method(docker, [
    registry('registry.example.com'),
    image_prefix('unifyweaver/'),
    orchestrator(kubernetes)  % or docker_compose, swarm
]).
```

#### 1.3 Cloud Functions (Future)

```prolog
:- deploy_method(cloud_function, [
    provider(aws_lambda),  % or gcp_functions, azure_functions
    region('us-east-1')
]).
```

### Service Declarations

#### Basic Service Definition

```prolog
:- remote_service(ml_predictor, [
    % Deployment
    host('ml.example.com'),
    deploy_method(ssh),

    % Runtime
    target(python),
    entry_point('predictor.py'),
    port(8080),

    % Lifecycle
    lifecycle(persistent),  % or transient, on_demand

    % Security
    transport(https),       % Required for remote (default)
    auth(bearer_token)
]).
```

#### Lifecycle Types

| Type | Description | Use Case |
|------|-------------|----------|
| `persistent` | Stays running indefinitely | Production services |
| `transient` | Shuts down after idle timeout | Batch processing |
| `on_demand` | Starts on first request, shuts down after | Cost optimization |
| `pipeline_bound` | Lives for duration of pipeline | One-off processing |

```prolog
:- remote_service(batch_processor, [
    lifecycle(transient),
    idle_timeout(300),          % Shutdown after 5 min idle
    max_lifetime(3600)          % Force shutdown after 1 hour
]).

:- remote_service(etl_worker, [
    lifecycle(pipeline_bound),
    shutdown_on(pipe_closed),   % Shutdown when stdin closes
    shutdown_on(signal(term))   % Or on SIGTERM
]).
```

### Change Detection and Redeployment

#### Source Tracking

```prolog
:- service_sources(ml_predictor, [
    'src/ml/predictor.py',
    'src/ml/models/*.pkl',
    'requirements.txt'
]).
```

#### Change Detection Methods

```prolog
:- change_detection(ml_predictor, [
    method(content_hash),       % SHA256 of source files (default)
    % method(git_commit),       % Compare git HEAD
    % method(mtime),            % File modification time
    check_interval(60)          % Check every 60 seconds
]).
```

#### Redeployment Hooks

```prolog
:- on_change(ml_predictor, [
    pre_shutdown(drain_connections),    % Graceful shutdown
    pre_shutdown(save_state),           % Persist any state
    post_deploy(health_check),          % Verify service is up
    post_deploy(warm_cache),            % Pre-warm if needed
    rollback_on_failure(true)           % Revert if health check fails
]).
```

### Security Requirements

#### Encryption Policy

```prolog
%% Security defaults - cannot be disabled for remote services
:- security_policy([
    remote_requires_encryption(true),   % SSH, HTTPS, or VPN
    local_allows_plaintext(true),       % localhost can use HTTP
    min_tls_version('1.2'),
    allowed_ciphers([aes256_gcm, chacha20_poly1305])
]).
```

#### Transport Selection

| Location | Default Transport | Encryption |
|----------|------------------|------------|
| `localhost` | HTTP | Optional |
| `local_network` | HTTPS | Required |
| `remote` | HTTPS | Required |
| `vpn` | HTTP | VPN provides encryption |

```prolog
%% Override for trusted network
:- network_zone(internal, [
    hosts(['10.0.0.0/8', '192.168.0.0/16']),
    encryption(optional),       % Internal network trusted
    transport(http)
]).
```

#### Authentication Methods

```prolog
:- service_auth(ml_predictor, [
    method(bearer_token),
    token_source(env('ML_API_TOKEN')),  % From environment
    % token_source(file('~/.tokens/ml')),
    % token_source(vault('secret/ml-token')),
    token_rotation(3600)                % Rotate hourly
]).

:- service_auth(internal_service, [
    method(mtls),                       % Mutual TLS
    client_cert('certs/client.pem'),
    client_key('certs/client-key.pem')
]).
```

### Deployment API

#### Prolog Predicates

```prolog
%% Deploy a service
deploy_service(+ServiceName, +Options) → success | error(Reason)

%% Check deployment status
service_status(+ServiceName, -Status) →
    Status = running(Pid, Since) | stopped | deploying | error(Msg)

%% Trigger redeployment
redeploy_service(+ServiceName, +Options) → success | error(Reason)

%% Shutdown service
shutdown_service(+ServiceName, +Options) → success | error(Reason)
    Options: [graceful(true), timeout(30), force(false)]

%% Check for changes
check_service_changes(+ServiceName, -Changes) →
    Changes = [file(Path, OldHash, NewHash), ...]
```

#### Generated Deployment Scripts

For SSH deployment, generate shell scripts:

```bash
#!/bin/bash
# deploy_ml_predictor.sh - Generated by UnifyWeaver

set -euo pipefail

SERVICE="ml_predictor"
HOST="ml.example.com"
USER="deploy"
REMOTE_DIR="/opt/unifyweaver/services/ml_predictor"

# Check SSH agent
if ! ssh-add -l &>/dev/null; then
    echo "Error: SSH agent not running or no keys loaded"
    exit 1
fi

# Create remote directory
ssh "${USER}@${HOST}" "mkdir -p ${REMOTE_DIR}"

# Sync sources
rsync -avz --delete \
    --exclude '*.pyc' \
    --exclude '__pycache__' \
    src/ml/ "${USER}@${HOST}:${REMOTE_DIR}/"

# Install dependencies
ssh "${USER}@${HOST}" "cd ${REMOTE_DIR} && pip install -r requirements.txt"

# Restart service
ssh "${USER}@${HOST}" "systemctl --user restart ${SERVICE}"

# Health check
for i in {1..30}; do
    if curl -sf "https://${HOST}:8080/health" >/dev/null; then
        echo "Service healthy"
        exit 0
    fi
    sleep 1
done

echo "Health check failed"
exit 1
```

---

## Part 2: Error Handling

### Retry Policies

```prolog
:- error_policy(ml_predictor, [
    on_error(retry),
    max_retries(3),
    retry_delay(exponential(1000, 2, 30000)),  % 1s, 2s, 4s... max 30s
    retry_on([timeout, connection_refused, 503]),
    fail_on([400, 401, 403, 404])              % Don't retry client errors
]).
```

### Fallback Mechanisms

```prolog
:- fallback(ml_predictor, [
    % Try backup service
    fallback_service(ml_predictor_backup),

    % Or use cached result
    fallback_cache(ttl(3600)),

    % Or use default value
    fallback_default({status: "unavailable", cached: true}),

    % Or call alternative predicate
    fallback_predicate(simple_predictor/2)
]).
```

### Circuit Breaker

```prolog
:- circuit_breaker(ml_predictor, [
    failure_threshold(5),           % Open after 5 failures
    success_threshold(3),           % Close after 3 successes
    half_open_timeout(30),          % Try again after 30s
    window(60)                      % Sliding window of 60s
]).
```

### Timeout Configuration

```prolog
:- timeouts(ml_predictor, [
    connect_timeout(5),             % 5s to establish connection
    read_timeout(30),               % 30s for response
    total_timeout(60),              % 60s end-to-end
    idle_timeout(300)               % Close idle connections after 5min
]).
```

---

## Part 3: Monitoring

### Health Checks

```prolog
:- health_check(ml_predictor, [
    endpoint('/health'),
    interval(30),                   % Check every 30s
    timeout(5),
    unhealthy_threshold(3),         % Mark unhealthy after 3 failures
    healthy_threshold(2)            % Mark healthy after 2 successes
]).
```

### Metrics Collection

```prolog
:- metrics(ml_predictor, [
    collect([
        request_count,
        request_latency_histogram,
        error_count,
        active_connections
    ]),
    labels([service, method, status_code]),
    export(prometheus),             % or statsd, cloudwatch
    port(9090)
]).
```

### Logging

```prolog
:- logging(ml_predictor, [
    level(info),                    % debug, info, warn, error
    format(json),                   % or text
    output(stdout),                 % or file('/var/log/service.log')
    include([timestamp, request_id, latency, status])
]).
```

### Alerting

```prolog
:- alerts(ml_predictor, [
    alert(high_error_rate, [
        condition('error_rate > 0.05'),  % >5% errors
        duration(300),                    % For 5 minutes
        severity(critical),
        notify([slack('#alerts'), pagerduty])
    ]),
    alert(high_latency, [
        condition('p99_latency > 1000'),  % >1s p99
        duration(60),
        severity(warning),
        notify([slack('#monitoring')])
    ])
]).
```

---

## Part 4: Implementation Plan

### Phase 6a: Deployment Foundation
- [ ] SSH deployment generator
- [ ] Service declaration parser
- [ ] Lifecycle management (start/stop/restart)
- [ ] Basic change detection (content hash)
- [ ] Security validation (encryption checks)

### Phase 6b: Advanced Deployment
- [ ] Pre/post deployment hooks
- [ ] Rollback support
- [ ] Multi-host deployment
- [ ] Health check integration
- [ ] Graceful shutdown

### Phase 6c: Error Handling
- [ ] Retry policies with backoff
- [ ] Fallback mechanisms
- [ ] Circuit breaker pattern
- [ ] Timeout configuration

### Phase 6d: Monitoring
- [ ] Health check generator
- [ ] Metrics collection (Prometheus)
- [ ] Structured logging
- [ ] Alert definitions

---

## Security Considerations

### SSH Agent Security

1. **Agent Forwarding Risk**: Only forward to trusted hosts
2. **Key Scope**: Use deployment-specific keys with limited permissions
3. **Timeout**: Set `AddKeysToAgent` with timeout

```bash
# Recommended SSH config
Host deploy-*
    ForwardAgent yes
    AddKeysToAgent 1h
    IdentityFile ~/.ssh/deploy_key
```

### Network Security

1. **No Plaintext Remote**: HTTP only allowed for localhost
2. **Certificate Validation**: Always verify TLS certificates
3. **Credential Storage**: Never store tokens in code; use env/vault

### Deployment Security

1. **Least Privilege**: Deploy user has minimal permissions
2. **Audit Trail**: Log all deployments with user, time, changes
3. **Approval Gates**: Optional manual approval for production

---

## Example: Complete Service Setup

```prolog
%% Define the ML prediction service
:- module(ml_service_config, []).

:- use_module('src/unifyweaver/glue/deployment_glue').

%% Deployment configuration
:- deploy_method(ml_predictor, ssh, [
    host('ml.example.com'),
    user('deploy'),
    agent(true),
    remote_dir('/opt/services/ml')
]).

%% Service definition
:- remote_service(ml_predictor, [
    target(python),
    entry_point('server.py'),
    port(8080),
    lifecycle(persistent),
    transport(https),
    cert('/etc/ssl/ml.pem')
]).

%% Source tracking
:- service_sources(ml_predictor, [
    'src/ml/**/*.py',
    'models/*.pkl',
    'requirements.txt'
]).

%% Change handling
:- on_change(ml_predictor, [
    pre_shutdown(drain_connections, [timeout(30)]),
    post_deploy(health_check, [retries(5)]),
    rollback_on_failure(true)
]).

%% Error handling
:- error_policy(ml_predictor, [
    max_retries(3),
    retry_delay(exponential(1000, 2, 10000)),
    fallback_service(ml_predictor_backup)
]).

%% Monitoring
:- health_check(ml_predictor, [
    endpoint('/health'),
    interval(30)
]).

:- metrics(ml_predictor, [
    export(prometheus),
    port(9090)
]).

%% Generate deployment artifacts
generate_all :-
    generate_deploy_script(ml_predictor, 'deploy/ml_predictor.sh'),
    generate_systemd_unit(ml_predictor, 'deploy/ml_predictor.service'),
    generate_health_check(ml_predictor, 'deploy/health_check.sh').
```

---

## Phase 7: Cloud & Enterprise Features (Future)

The following advanced features are deferred to Phase 7:

### Container Deployment
- Docker image building and registry push
- Kubernetes deployment manifests
- Docker Compose generation
- Container orchestration (Swarm, ECS)

### Secret Management Integration
- HashiCorp Vault integration
- AWS Secrets Manager
- Azure Key Vault
- GCP Secret Manager

### Multi-Region & Geographic Distribution
- Region-aware service routing
- Cross-region replication
- Latency-based load balancing
- Geo-failover

### Stateful Service Handling
- State persistence during redeployment
- Database migration coordination
- Session draining strategies
- Persistent volume management

### Cloud Functions
- AWS Lambda deployment
- GCP Cloud Functions
- Azure Functions
- Serverless framework integration

---

## Design Decisions

1. **Rollback Scope**: Automatic rollback triggers on health check failure only. Other failures (e.g., file sync errors) halt deployment but don't rollback.

2. **SSH as Primary**: SSH deployment is the foundation because it works everywhere, requires minimal infrastructure, and provides strong security via agent forwarding.

3. **Security Non-Negotiable**: Remote encryption requirement cannot be disabled. This is a safety rail, not a preference.

---

## Related Documents

- [03-implementation-plan.md](03-implementation-plan.md) - Overall phase status
- [04-api-reference.md](04-api-reference.md) - Existing API documentation
- [Network Glue](../../guides/cross-target-glue.md#network-glue) - Current network layer
