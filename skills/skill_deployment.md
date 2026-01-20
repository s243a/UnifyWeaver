# Skill: Deployment

Automatic deployment and lifecycle management for remote services including Docker, Kubernetes, cloud functions, and secrets management.

## When to Use

- User asks "how do I deploy my service?"
- User needs Docker or Kubernetes manifests
- User wants AWS Lambda, Google Cloud Functions, or Azure Functions
- User needs secrets management (Vault, AWS, Azure, GCP)

## Quick Start

```prolog
:- use_module('src/unifyweaver/glue/deployment_glue').

% Declare service
declare_service(my_api, [
    host('server.example.com'),
    port(8080),
    target(python),
    entry_point('main.py'),
    lifecycle(persistent)
]).

% Generate Dockerfile
generate_dockerfile(my_api, [], Dockerfile).

% Deploy
deploy_service(my_api, Result).
```

## Service Declaration

### Declare Service

```prolog
declare_service(Name, Options).
```

**Options:**
- `host(Host)` - Remote hostname or IP
- `port(Port)` - Service port (default: 8080)
- `target(Target)` - Compilation target (python, go, rust)
- `entry_point(File)` - Main file to execute
- `lifecycle(Type)` - persistent | transient | on_demand | pipeline_bound
- `transport(T)` - http | https | ssh
- `idle_timeout(Secs)` - For transient services
- `max_lifetime(Secs)` - Maximum runtime

### Service Lifecycle Types

| Lifecycle | Description |
|-----------|-------------|
| `persistent` | Always running |
| `transient` | Stops after idle timeout |
| `on_demand` | Starts on request |
| `pipeline_bound` | Lives for pipeline duration |

## SSH Deployment

### Generate Deploy Script

```prolog
generate_deploy_script(Service, Options, Script).
generate_ssh_deploy(Service, Options, Script).
```

### Multi-Host Deployment

```prolog
declare_service_hosts(my_api, ['host1.example.com', 'host2.example.com']).
deploy_to_all_hosts(my_api, Results).
```

### Rollback Support

```prolog
deploy_with_rollback(my_api, Result).
rollback_service(my_api, Result).
```

## Docker Deployment

### Generate Dockerfile

```prolog
generate_dockerfile(Service, Options, Dockerfile).
```

### Generate .dockerignore

```prolog
generate_dockerignore(Service, Options, Content).
```

### Build and Push

```prolog
build_docker_image(my_api, [tag('v1.0')], Result).
push_docker_image(my_api, [registry(my_registry)], Result).
```

### Docker Compose

```prolog
declare_compose_config(my_project, [
    services([my_api, my_db]),
    networks([backend]),
    volumes([data])
]).

generate_docker_compose(my_project, [], ComposeYaml).
```

## Kubernetes Deployment

### Declare K8s Config

```prolog
declare_k8s_config(my_api, [
    replicas(3),
    namespace(production),
    image('myregistry/myapi:latest'),
    ports([8080]),
    resources([
        requests([cpu('100m'), memory('128Mi')]),
        limits([cpu('500m'), memory('512Mi')])
    ])
]).
```

### Generate Manifests

```prolog
% Deployment
generate_k8s_deployment(my_api, [], Manifest).

% Service
generate_k8s_service(my_api, [], ServiceManifest).

% ConfigMap
generate_k8s_configmap(my_api, [], ConfigMapManifest).

% Ingress
generate_k8s_ingress(my_api, [], IngressManifest).

% Helm Chart
generate_helm_chart(my_api, [], Chart).
```

### Deploy to Kubernetes

```prolog
deploy_to_k8s(my_api, Options, Result).
scale_k8s_deployment(my_api, 5, Options, Result).
rollout_status(my_api, Options, Status).
```

## Cloud Functions (EXPERIMENTAL)

### AWS Lambda

```prolog
declare_lambda_config(my_func, [
    runtime(python39),
    handler('main.handler'),
    memory(256),
    timeout(30)
]).

generate_lambda_function(my_func, [], Package).
generate_lambda_deploy(my_func, [], Commands).
generate_sam_template(my_func, [], Template).
```

### Google Cloud Functions

```prolog
declare_gcf_config(my_func, [
    runtime(python39),
    entry_point(handler),
    memory(256)
]).

generate_gcf_deploy(my_func, [], Commands).
```

### Azure Functions

```prolog
declare_azure_func_config(my_func, [
    runtime(python),
    plan(consumption)
]).

generate_azure_func_deploy(my_func, [], Commands).
```

### API Gateway Integration

```prolog
declare_api_gateway(my_gateway, [
    provider(aws),
    routes([
        route('/api/data', my_func_data),
        route('/api/process', my_func_process)
    ])
]).

generate_api_gateway_config(my_gateway, [], Config).
generate_openapi_spec(my_gateway, [], Spec).
```

## Secrets Management (EXPERIMENTAL)

### HashiCorp Vault

```prolog
declare_vault_config(my_vault, [
    addr('https://vault.example.com'),
    auth_method(token)
]).

generate_vault_read(my_vault, 'secret/data/myapp', [], Command).
generate_vault_agent_config(my_api, [], Config).
```

### AWS Secrets Manager

```prolog
declare_aws_secrets_config(aws_secrets, [region('us-west-2')]).
generate_aws_secret_read(aws_secrets, 'prod/myapp/db', [], Command).
```

### Azure Key Vault

```prolog
declare_azure_keyvault_config(azure_kv, [vault_name('myvault')]).
generate_azure_secret_read(azure_kv, 'db-password', [], Command).
```

### GCP Secret Manager

```prolog
declare_gcp_secrets_config(gcp_secrets, [project('my-project')]).
generate_gcp_secret_read(gcp_secrets, 'api-key', [], Command).
```

### Service Secret Bindings

```prolog
declare_service_secrets(my_api, [
    secret(db_password, vault, 'secret/data/db'),
    secret(api_key, aws, 'prod/api-key')
]).

generate_secret_env_script(my_api, [], Script).
generate_k8s_secret(my_api, [], Manifest).
generate_k8s_external_secret(my_api, [], Manifest).
```

## Multi-Region Deployment (EXPERIMENTAL)

### Declare Regions

```prolog
declare_region(us_west, [
    provider(aws),
    zone('us-west-2'),
    endpoints(['api-west.example.com'])
]).

declare_service_regions(my_api, [
    primary(us_west),
    secondary([us_east, eu_west])
]).
```

### Failover Policy

```prolog
declare_failover_policy(my_api, [
    strategy(health_based),
    health_check_interval(30),
    failover_threshold(3)
]).

select_region(my_api, [], Region).
failover_to_region(my_api, us_east, Result).
```

### Multi-Region Deploy

```prolog
deploy_to_region(my_api, us_west, [], Result).
deploy_to_all_regions(my_api, [], Results).
```

### Traffic Management

```prolog
declare_traffic_policy(my_api, [
    routing(weighted),
    weights([us_west: 70, us_east: 30])
]).

generate_route53_config(my_api, [], Config).
generate_cloudflare_config(my_api, [], Config).
```

## Lifecycle Operations

```prolog
start_service(my_api, Result).
stop_service(my_api, Result).
restart_service(my_api, Result).
graceful_stop(my_api, [timeout(30)], Result).
```

## Health Checks and Monitoring

### Health Checks

```prolog
declare_health_check(my_api, [
    endpoint('/health'),
    interval(30),
    timeout(5),
    healthy_threshold(2),
    unhealthy_threshold(3)
]).

run_health_check(my_api, [], Result).
wait_for_healthy(my_api, [timeout(60)], Result).
```

### Metrics

```prolog
declare_metrics(my_api, [
    prometheus(true),
    metrics([requests_total, latency_seconds])
]).

record_metric(my_api, requests_total, 1).
get_metrics(my_api, Metrics).
generate_prometheus_metrics(my_api, Output).
```

### Alerting

```prolog
declare_alert(my_api, high_latency, [
    condition(latency_p99 > 1000),
    severity(warning)
]).

check_alerts(my_api, TriggeredAlerts).
```

## Error Handling

### Retry Policies

```prolog
declare_retry_policy(my_api, [
    max_retries(3),
    backoff(exponential),
    initial_delay(1000)
]).

call_with_retry(my_api, operation, Args, Result).
```

### Circuit Breaker

```prolog
declare_circuit_breaker(my_api, [
    failure_threshold(5),
    recovery_timeout(30)
]).

call_with_circuit_breaker(my_api, operation, Args, Result).
reset_circuit_breaker(my_api).
```

## Related

**Parent Skill:**
- `skill_infrastructure.md` - Infrastructure sub-master

**Sibling Skills:**
- `skill_authentication.md` - Auth backends
- `skill_networking.md` - HTTP/socket generation

**Code:**
- `src/unifyweaver/glue/deployment_glue.pl`
