<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 3: Lifecycle Hooks

Lifecycle hooks provide runtime security checkpoints that execute at critical moments during service deployment and operation. This chapter covers how to declare and use hooks effectively.

## What Are Lifecycle Hooks?

Lifecycle hooks are actions that execute automatically at specific events:

```
┌─────────────────────────────────────────────────────────────┐
│                    SERVICE LIFECYCLE                         │
│                                                              │
│  pre_deploy → Deploy → post_deploy → Running → pre_shutdown │
│      │                      │           │            │       │
│      ↓                      ↓           ↓            ↓       │
│  validate               health_check  monitoring    drain    │
│                         warm_cache                  cleanup  │
└─────────────────────────────────────────────────────────────┘
```

## Declaring Hooks

Use `declare_lifecycle_hook/3` to register hooks:

```prolog
:- declare_lifecycle_hook(ServiceName, Event, Action).
```

**Example**:

```prolog
% Register hooks for api_service
:- declare_lifecycle_hook(api_service, pre_deploy, health_check).
:- declare_lifecycle_hook(api_service, post_deploy, warm_cache).
:- declare_lifecycle_hook(api_service, pre_shutdown, drain_connections).
:- declare_lifecycle_hook(api_service, on_health_failure, custom('alert.sh')).
```

## Hook Events

### pre_deploy

Executes before deployment begins:

```prolog
:- declare_lifecycle_hook(my_service, pre_deploy, custom('validate.sh')).
```

**Use cases**:
- Validate configuration
- Check prerequisites
- Backup current state
- Notify team

### post_deploy

Executes after deployment completes:

```prolog
:- declare_lifecycle_hook(my_service, post_deploy, health_check).
:- declare_lifecycle_hook(my_service, post_deploy, warm_cache).
```

**Use cases**:
- Verify service is healthy
- Warm up caches
- Run smoke tests
- Send deployment notification

### pre_shutdown

Executes before stopping the service:

```prolog
:- declare_lifecycle_hook(my_service, pre_shutdown, drain_connections).
:- declare_lifecycle_hook(my_service, pre_shutdown, save_state).
```

**Use cases**:
- Drain active connections
- Save service state
- Complete pending work
- Notify load balancer

### post_shutdown

Executes after service stops:

```prolog
:- declare_lifecycle_hook(my_service, post_shutdown, custom('cleanup.sh')).
```

**Use cases**:
- Clean up temporary files
- Release resources
- Archive logs
- Notify monitoring

### on_health_failure

Executes when health check fails:

```prolog
:- declare_lifecycle_hook(my_service, on_health_failure, custom('alert.sh')).
```

**Use cases**:
- Send alerts
- Trigger auto-recovery
- Log diagnostics
- Failover to backup

## Built-in Hook Actions

### drain_connections

Wait for active connections to complete:

```prolog
:- declare_lifecycle_hook(api_service, pre_shutdown, drain_connections).
```

**Generated code**:
```bash
echo "Draining connections..."
ssh "deploy@host" "sleep 5"  # Allow connections to drain
```

### health_check

Verify service is responding:

```prolog
:- declare_lifecycle_hook(api_service, post_deploy, health_check).
```

**Generated code**:
```bash
echo "Running health check..."
for i in {1..30}; do
    if curl -sf "http://host:${PORT}/health" >/dev/null; then
        echo "Health check passed"
        break
    fi
    sleep 1
done
```

### warm_cache

Pre-populate caches:

```prolog
:- declare_lifecycle_hook(api_service, post_deploy, warm_cache).
```

**Generated code**:
```bash
echo "Warming cache..."
ssh "deploy@host" "curl -s http://localhost:${PORT}/warmup || true"
```

### save_state

Persist service state before shutdown:

```prolog
:- declare_lifecycle_hook(api_service, pre_shutdown, save_state).
```

**Generated code**:
```bash
echo "Saving state..."
ssh "deploy@host" "cd ${REMOTE_DIR} && ./save_state.sh || true"
```

### custom(Command)

Run arbitrary command:

```prolog
:- declare_lifecycle_hook(api_service, post_deploy, custom('notify.sh deployed')).
:- declare_lifecycle_hook(api_service, on_health_failure, custom('pagerduty-alert.sh')).
```

**Generated code**:
```bash
echo "Running custom hook..."
ssh "deploy@host" "notify.sh deployed"
```

## Hook Execution Flow

### deploy_with_hooks/2

The main deployment predicate executes hooks automatically:

```prolog
deploy_with_hooks(Service, Result) :-
    % 1. Validate security
    validate_security(Service, SecurityErrors),
    (SecurityErrors \== []
    ->  Result = error(security_validation_failed(SecurityErrors))

    % 2. Execute pre-deploy hooks
    ;   execute_hooks(Service, pre_deploy, PreResult),
        (PreResult \== ok
        ->  Result = error(pre_deploy_failed(PreResult))

        % 3. Perform deployment
        ;   do_deployment(Service, DeployResult),
            (DeployResult \== deployed
            ->  Result = error(deployment_failed(DeployResult))

            % 4. Execute post-deploy hooks
            ;   execute_hooks(Service, post_deploy, PostResult),
                (PostResult == ok
                ->  Result = deployed
                ;   Result = deployed_with_warnings(PostResult)
                )
            )
        )
    ).
```

### graceful_stop/3

Graceful shutdown with hooks:

```prolog
graceful_stop(Service, Options, Result) :-
    option_or_default(drain_timeout, Options, 30, DrainTimeout),

    % 1. Execute pre-shutdown hooks
    execute_hooks(Service, pre_shutdown, PreResult),
    (PreResult == ok
    ->  % 2. Drain connections
        drain_connections(Service, [timeout(DrainTimeout)], DrainResult),
        (DrainResult == drained
        ->  % 3. Stop service
            stop_service(Service, StopResult),
            Result = StopResult
        ;   Result = error(drain_failed(DrainResult))
        )
    ;   Result = error(pre_shutdown_hook_failed(PreResult))
    ).
```

## Multiple Hooks per Event

You can declare multiple hooks for the same event:

```prolog
% Multiple post-deploy hooks - executed in order
:- declare_lifecycle_hook(api_service, post_deploy, health_check).
:- declare_lifecycle_hook(api_service, post_deploy, warm_cache).
:- declare_lifecycle_hook(api_service, post_deploy, custom('notify.sh')).
```

Hooks execute sequentially. If any hook fails, subsequent hooks are skipped.

## Hook Error Handling

### Hook Failures

If a hook fails:

```prolog
execute_hook_actions(Service, [Action|Rest], Result) :-
    execute_single_hook(Service, Action, ActionResult),
    (ActionResult == ok
    ->  execute_hook_actions(Service, Rest, Result)
    ;   Result = error(hook_failed(Action, ActionResult))
    ).
```

### Ignoring Hook Failures

For non-critical hooks, wrap in custom with error handling:

```prolog
% Continue even if notification fails
:- declare_lifecycle_hook(api_service, post_deploy,
    custom('notify.sh || true')).
```

## Querying Hooks

### List All Hooks for a Service

```prolog
?- lifecycle_hooks(api_service, Hooks).
Hooks = [
    hook(pre_deploy, health_check),
    hook(post_deploy, warm_cache),
    hook(pre_shutdown, drain_connections)
].
```

### Check if Hook Exists

```prolog
?- lifecycle_hook_db(api_service, post_deploy, _).
true.
```

## Complete Example

Service with comprehensive lifecycle management:

```prolog
% service_setup.pl

% 1. Declare service
:- declare_service(analytics_api, [
    host('analytics.example.com'),
    port(8080),
    target(go),
    transport(https)
]).

% 2. Configure deployment
:- declare_deploy_method(analytics_api, ssh, [
    user('deploy'),
    remote_dir('/opt/analytics')
]).

% 3. Declare lifecycle hooks
%    Pre-deploy: Validate configuration
:- declare_lifecycle_hook(analytics_api, pre_deploy,
    custom('./scripts/validate-config.sh')).

%    Post-deploy: Health check then warm caches
:- declare_lifecycle_hook(analytics_api, post_deploy, health_check).
:- declare_lifecycle_hook(analytics_api, post_deploy, warm_cache).
:- declare_lifecycle_hook(analytics_api, post_deploy,
    custom('./scripts/notify-slack.sh deployed')).

%    Pre-shutdown: Graceful drain
:- declare_lifecycle_hook(analytics_api, pre_shutdown, drain_connections).
:- declare_lifecycle_hook(analytics_api, pre_shutdown, save_state).

%    Post-shutdown: Cleanup
:- declare_lifecycle_hook(analytics_api, post_shutdown,
    custom('./scripts/archive-logs.sh')).

%    Health failure: Alert
:- declare_lifecycle_hook(analytics_api, on_health_failure,
    custom('./scripts/pagerduty-alert.sh')).

% 4. Configure error resilience
:- declare_retry_policy(analytics_api, [
    max_retries(3),
    retry_delay(exponential(1000, 2, 10000))
]).

:- declare_circuit_breaker(analytics_api, [
    failure_threshold(5),
    reset_timeout(60)
]).

% 5. Configure health checks
:- declare_health_check(analytics_api, [
    endpoint('/health'),
    interval(30),
    timeout(5),
    healthy_threshold(2),
    unhealthy_threshold(3)
]).

% Deploy with full lifecycle
deploy_analytics :-
    format('Deploying analytics_api...~n'),
    deploy_with_hooks(analytics_api, Result),
    format('Result: ~w~n', [Result]).

% Graceful shutdown
shutdown_analytics :-
    format('Shutting down analytics_api...~n'),
    graceful_stop(analytics_api, [drain_timeout(30)], Result),
    format('Result: ~w~n', [Result]).
```

## Generated Deployment Script

The hooks are compiled into the deployment script:

```bash
#!/bin/bash
set -euo pipefail

SERVICE=analytics_api
HOST=analytics.example.com
USER=deploy
REMOTE_DIR=/opt/analytics
PORT=8080

# Pre-shutdown hooks
echo "Running pre-shutdown hooks..."
echo "Draining connections..."
ssh "${USER}@${HOST}" "sleep 5"
echo "Saving state..."
ssh "${USER}@${HOST}" "cd ${REMOTE_DIR} && ./save_state.sh || true"

# Stop existing service
ssh "${USER}@${HOST}" "pkill -f 'analytics' || true"

# Deploy new version
rsync -avz --delete ./dist/ "${USER}@${HOST}:${REMOTE_DIR}/"

# Start service
ssh "${USER}@${HOST}" "cd ${REMOTE_DIR} && ./analytics -port=${PORT} &"

# Post-deploy hooks
echo "Running post-deploy hooks..."
echo "Running health check..."
for i in {1..30}; do
    if curl -sf "http://${HOST}:${PORT}/health" >/dev/null; then
        echo "Health check passed"
        break
    fi
    sleep 1
done
echo "Warming cache..."
ssh "${USER}@${HOST}" "curl -s http://localhost:${PORT}/warmup || true"
echo "Running custom hook..."
ssh "${USER}@${HOST}" "./scripts/notify-slack.sh deployed"

echo "Deployment complete: ${SERVICE}"
```

## Security Considerations

### Hook Command Injection

Custom hooks execute arbitrary commands. Validate inputs:

```prolog
% DANGEROUS - User input in command
:- declare_lifecycle_hook(svc, post_deploy, custom(UserInput)).

% SAFE - Predefined commands only
:- declare_lifecycle_hook(svc, post_deploy, custom('predefined-script.sh')).
```

### Hook Permissions

Ensure hook scripts have appropriate permissions:

```bash
# Restrict hook script access
chmod 700 scripts/*.sh
chown deploy:deploy scripts/*.sh
```

### Hook Timeouts

Set timeouts to prevent hanging hooks:

```prolog
:- declare_lifecycle_hook(svc, post_deploy,
    custom('timeout 30 slow-script.sh')).
```

## Summary

Lifecycle hooks provide:

- **Deployment Safety**: Validate before, verify after
- **Graceful Operations**: Drain connections, save state
- **Automated Recovery**: Health check triggers, alerts
- **Flexibility**: Custom commands for any need

Key points:
- Declare hooks with `declare_lifecycle_hook/3`
- Built-in actions: `drain_connections`, `health_check`, `warm_cache`, `save_state`
- Custom actions: `custom(Command)` for arbitrary scripts
- Multiple hooks per event execute in order
- Hook failures can abort the operation

The next chapter covers target-specific security considerations.

---

## Navigation

**←** [Previous: Chapter 2: Firewall Policies](02_firewall_policies) | [📖 Book 8: Security & Firewall](./) | [Next: Chapter 4: Target Security →](04_target_security)
