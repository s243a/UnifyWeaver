<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 6: Production Hardening

This chapter covers production-grade security measures: encryption, secrets management, monitoring, and operational security.

## Transport Security

### HTTPS Enforcement

Remote services must use encrypted transport:

```prolog
% Service declaration with HTTPS
:- declare_service(api_service, [
    host('api.example.com'),
    port(8443),
    transport(https),         % Required for remote
    tls_version('1.3'),       % Minimum TLS version
    cert_file('/etc/ssl/cert.pem'),
    key_file('/etc/ssl/key.pem')
]).

% Validation automatically enforces
validate_security(api_service, Errors).
% If transport != https for remote: Errors = [remote_requires_encryption]
```

### TLS Configuration

```prolog
:- declare_tls_config(production, [
    min_version('1.2'),
    preferred_version('1.3'),
    cipher_suites([
        'TLS_AES_256_GCM_SHA384',
        'TLS_CHACHA20_POLY1305_SHA256',
        'TLS_AES_128_GCM_SHA256'
    ]),
    verify_peer(true),
    verify_depth(3)
]).
```

### Certificate Management

```prolog
% Declare certificate configuration
:- declare_cert_config(api_service, [
    cert_source(acme),          % Let's Encrypt
    domain('api.example.com'),
    renewal_days(30),           % Renew 30 days before expiry
    on_renewal(restart_service)
]).

% Or use existing certificates
:- declare_cert_config(api_service, [
    cert_source(file),
    cert_file('/etc/ssl/certs/api.crt'),
    key_file('/etc/ssl/private/api.key'),
    ca_file('/etc/ssl/certs/ca.crt')
]).
```

## Secrets Management

### Environment Variables

Basic secret injection:

```prolog
% Declare secrets from environment
:- declare_secret(api_service, database_url, [
    source(env('DATABASE_URL')),
    required(true)
]).

:- declare_secret(api_service, api_key, [
    source(env('API_KEY')),
    required(true),
    redact_in_logs(true)
]).
```

### External Secret Stores

Integration with HashiCorp Vault, AWS, Azure, GCP:

```prolog
% HashiCorp Vault
:- declare_secret(api_service, db_password, [
    source(vault('secret/data/database', 'password')),
    vault_addr('https://vault.example.com:8200'),
    vault_auth(kubernetes)
]).

% AWS Secrets Manager
:- declare_secret(api_service, api_key, [
    source(aws_secrets('prod/api-key')),
    region('us-west-2')
]).

% Azure Key Vault
:- declare_secret(api_service, connection_string, [
    source(azure_keyvault('prod-vault', 'db-connection')),
    managed_identity(true)
]).

% GCP Secret Manager
:- declare_secret(api_service, credentials, [
    source(gcp_secrets('projects/myproj/secrets/creds/versions/latest'))
]).
```

### Secret Rotation

```prolog
% Configure automatic rotation
:- declare_secret_rotation(api_service, api_key, [
    interval('90d'),
    notify([email, slack]),
    pre_rotate(custom('backup-key.sh')),
    post_rotate(custom('verify-key.sh'))
]).
```

## Monitoring and Observability

### Health Checks

```prolog
:- declare_health_check(api_service, [
    endpoint('/health'),
    interval(30),              % Check every 30 seconds
    timeout(5),                % 5 second timeout
    healthy_threshold(2),      % 2 consecutive passes = healthy
    unhealthy_threshold(3)     % 3 consecutive fails = unhealthy
]).

% Liveness vs readiness
:- declare_health_check(api_service, liveness, [
    endpoint('/health/live'),
    interval(10)
]).

:- declare_health_check(api_service, readiness, [
    endpoint('/health/ready'),
    interval(5)
]).
```

### Metrics Collection

```prolog
:- declare_metrics(api_service, [
    format(prometheus),
    endpoint('/metrics'),
    include([
        requests_total,
        request_duration_seconds,
        errors_total,
        active_connections
    ])
]).
```

### Structured Logging

```prolog
:- declare_logging(api_service, [
    format(json),
    level(info),
    output(stdout),
    include_fields([
        timestamp,
        level,
        message,
        service,
        trace_id,
        span_id
    ]),
    redact([password, api_key, token])
]).
```

### Alerting

```prolog
:- declare_alert(api_service, high_error_rate, [
    condition(error_rate > 0.05),  % >5% errors
    duration('5m'),
    severity(critical),
    notify([pagerduty, slack])
]).

:- declare_alert(api_service, high_latency, [
    condition(p99_latency > 500),  % >500ms p99
    duration('10m'),
    severity(warning),
    notify([slack])
]).

:- declare_alert(api_service, unhealthy, [
    condition(health_status == unhealthy),
    duration('1m'),
    severity(critical),
    notify([pagerduty, email])
]).
```

## Resource Limits

### Memory and CPU

```prolog
:- declare_resource_limits(api_service, [
    memory_limit('512Mi'),
    memory_request('256Mi'),
    cpu_limit('1000m'),        % 1 CPU
    cpu_request('250m')        % 0.25 CPU
]).
```

### Connection Limits

```prolog
:- declare_connection_limits(api_service, [
    max_connections(1000),
    connection_timeout(30),
    idle_timeout(300),
    max_request_size('10Mi')
]).
```

### Rate Limiting

```prolog
:- declare_rate_limit(api_service, [
    requests_per_second(100),
    burst(150),
    by(client_ip),
    on_exceed(return_429)
]).
```

## Network Security

### Network Policies

```prolog
:- declare_network_policy(api_service, [
    allow_ingress([
        from(namespace('production')),
        from(cidr('10.0.0.0/8')),
        ports([8443])
    ]),
    allow_egress([
        to(namespace('database')),
        to(dns),
        ports([5432])
    ]),
    default_deny(true)
]).
```

### Service Mesh Integration

```prolog
:- declare_service_mesh(api_service, [
    mesh(istio),
    mtls(strict),
    circuit_breaker([
        consecutive_errors(5),
        interval('10s'),
        base_ejection_time('30s')
    ])
]).
```

## Complete Production Configuration

Example production-ready service:

```prolog
% production_config.pl - Complete production service

%% Service Definition
:- declare_service(analytics_api, [
    host('analytics.example.com'),
    port(8443),
    target(go),
    transport(https)
]).

%% Deployment
:- declare_deploy_method(analytics_api, kubernetes, [
    namespace('production'),
    replicas(3),
    strategy(rolling_update)
]).

%% TLS/Certificates
:- declare_cert_config(analytics_api, [
    cert_source(acme),
    domain('analytics.example.com'),
    renewal_days(30)
]).

%% Secrets
:- declare_secret(analytics_api, db_url, [
    source(vault('secret/data/analytics', 'db_url'))
]).
:- declare_secret(analytics_api, api_key, [
    source(env('ANALYTICS_API_KEY')),
    redact_in_logs(true)
]).

%% Health Checks
:- declare_health_check(analytics_api, liveness, [
    endpoint('/health/live'),
    interval(10),
    timeout(5)
]).
:- declare_health_check(analytics_api, readiness, [
    endpoint('/health/ready'),
    interval(5),
    timeout(3)
]).

%% Lifecycle Hooks
:- declare_lifecycle_hook(analytics_api, pre_deploy,
    custom('./scripts/validate.sh')).
:- declare_lifecycle_hook(analytics_api, post_deploy, health_check).
:- declare_lifecycle_hook(analytics_api, pre_shutdown, drain_connections).

%% Error Resilience
:- declare_retry_policy(analytics_api, [
    max_retries(3),
    retry_delay(exponential(1000, 2, 10000))
]).
:- declare_circuit_breaker(analytics_api, [
    failure_threshold(5),
    reset_timeout(60)
]).
:- declare_fallback(analytics_api, [
    default_value(#{status: unavailable})
]).

%% Resource Limits
:- declare_resource_limits(analytics_api, [
    memory_limit('1Gi'),
    cpu_limit('2000m')
]).
:- declare_rate_limit(analytics_api, [
    requests_per_second(1000),
    burst(1500)
]).

%% Monitoring
:- declare_metrics(analytics_api, [
    format(prometheus),
    endpoint('/metrics')
]).
:- declare_logging(analytics_api, [
    format(json),
    level(info)
]).

%% Alerting
:- declare_alert(analytics_api, high_error_rate, [
    condition(error_rate > 0.01),
    severity(critical),
    notify([pagerduty])
]).

%% Network Policy
:- declare_network_policy(analytics_api, [
    allow_ingress([from(namespace('gateway'))]),
    allow_egress([to(namespace('database'))]),
    default_deny(true)
]).

%% Deployment Procedure
deploy_production :-
    format('Deploying analytics_api to production...~n'),

    % 1. Validate security
    validate_security(analytics_api, SecurityErrors),
    (SecurityErrors \== []
    ->  format('Security validation failed: ~w~n', [SecurityErrors]),
        fail
    ;   true),

    % 2. Deploy with hooks
    deploy_with_hooks(analytics_api, Result),
    format('Deployment result: ~w~n', [Result]),

    % 3. Verify health
    wait_for_healthy(analytics_api, [timeout(120)], HealthResult),
    format('Health check: ~w~n', [HealthResult]).
```

## Security Checklist

Production deployment checklist:

### Transport Security
- [ ] HTTPS enabled for all remote services
- [ ] TLS 1.2+ enforced
- [ ] Strong cipher suites configured
- [ ] Certificates from trusted CA
- [ ] Automatic certificate renewal

### Authentication & Secrets
- [ ] Secrets stored in external vault
- [ ] No secrets in code or config files
- [ ] Secret rotation configured
- [ ] API authentication required
- [ ] Service-to-service mTLS

### Network Security
- [ ] Network policies defined
- [ ] Default deny enabled
- [ ] Egress restricted to required destinations
- [ ] Rate limiting configured

### Monitoring & Alerting
- [ ] Health checks configured
- [ ] Metrics exported
- [ ] Structured logging enabled
- [ ] Alerts for critical conditions
- [ ] On-call notification configured

### Error Handling
- [ ] Retry policies configured
- [ ] Circuit breakers enabled
- [ ] Fallbacks defined
- [ ] Graceful shutdown implemented

### Resource Management
- [ ] Memory/CPU limits set
- [ ] Connection limits configured
- [ ] Request size limits
- [ ] Timeout configured

## Summary

Production hardening encompasses:

**Transport Security**:
- HTTPS/TLS for all remote communication
- Certificate management and rotation
- Strong cipher configuration

**Secrets Management**:
- External secret stores (Vault, AWS, Azure, GCP)
- Environment variable injection
- Automatic rotation

**Monitoring**:
- Health checks (liveness, readiness)
- Prometheus metrics
- Structured JSON logging
- Alerting rules

**Resource Protection**:
- Memory/CPU limits
- Connection limits
- Rate limiting
- Network policies

This completes Book 8 on Security and Firewall.

---

## Navigation

**←** [Previous: Chapter 5: Validation & Fallback](05_validation_fallback) | [📖 Book 8: Security & Firewall](./) | [Next: Book 9: Rust Target →](../book-09-rust-target/)
