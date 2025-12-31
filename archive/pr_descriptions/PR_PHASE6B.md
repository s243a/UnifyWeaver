# Add Phase 6b: Advanced Deployment - Rollback, Multi-Host, Graceful Shutdown

## Summary

Extends the deployment glue module with Phase 6b advanced deployment features: automatic rollback on health check failure, multi-host deployment for redundancy, and graceful shutdown with connection draining.

## New Features

### Multi-Host Deployment

Deploy services to multiple hosts for load balancing and redundancy.

```prolog
% Configure multiple hosts
:- declare_service_hosts(api_service, [
    host_config('api1.example.com', [user('deploy')]),
    host_config('api2.example.com', [user('deploy')]),
    host_config('api3.example.com', [user('deploy')])
]).

% Deploy to all hosts
deploy_to_all_hosts(api_service, Results).
% Results = [result('api1.example.com', deployed),
%            result('api2.example.com', deployed), ...]
```

**New Predicates:**
- `declare_service_hosts/2` - Configure multiple deployment targets
- `service_hosts/2` - Query host configurations
- `deploy_to_all_hosts/2` - Deploy to all configured hosts

### Rollback Support

Automatic rollback when health checks fail after deployment.

```prolog
% Deploy with automatic rollback on failure
deploy_with_rollback(ml_service, Result).
% If health check fails: Result = rolled_back_after_failure(...)

% Manual rollback
rollback_service(ml_service, Result).
```

**Workflow:**
1. Store current deployment hash as rollback point
2. Create backup of current deployment
3. Execute deployment
4. Run health check
5. If unhealthy → automatically rollback to backup

**New Predicates:**
- `store_rollback_hash/2` - Save version for potential rollback
- `rollback_hash/2` - Query stored rollback version
- `rollback_service/2` - Execute rollback
- `deploy_with_rollback/2` - Deploy with automatic rollback
- `generate_rollback_script/3` - Generate rollback scripts

### Graceful Shutdown

Stop services gracefully with connection draining.

```prolog
% Graceful stop with draining
graceful_stop(api_service, [
    drain_timeout(30),    % Wait 30s for connections to drain
    force_after(60)       % Force kill after 60s
], Result).
```

**Workflow:**
1. Execute pre-shutdown hooks
2. Signal service to stop accepting new connections
3. Wait for active connections to complete (drain)
4. Stop service
5. If drain times out → force kill

**New Predicates:**
- `graceful_stop/3` - Stop with connection draining
- `drain_connections/3` - Wait for connections to complete
- `force_stop_service/3` - Force kill after timeout

### Hook Execution

Execute lifecycle hooks at deployment events.

```prolog
% Declare hooks
:- declare_lifecycle_hook(service, pre_shutdown, drain_connections).
:- declare_lifecycle_hook(service, pre_shutdown, save_state).
:- declare_lifecycle_hook(service, post_deploy, health_check).
:- declare_lifecycle_hook(service, post_deploy, warm_cache).

% Execute all hooks for an event
execute_hooks(service, pre_shutdown, Result).
```

**Supported Hook Actions:**
- `drain_connections` - Wait for active connections
- `health_check` - Verify service is healthy
- `save_state` - Persist service state
- `warm_cache` - Pre-warm caches
- `custom(Command)` - Run arbitrary command

### Health Check Integration

```prolog
% Run health check with retries
run_health_check(service, [
    retries(5),
    delay(2),
    timeout(5),
    endpoint('/health')
], Result).

% Wait for service to become healthy
wait_for_healthy(service, [timeout(60), interval(2)], Result).
```

### Deploy with Full Lifecycle

Complete deployment workflow with all hooks and safety checks.

```prolog
deploy_with_hooks(service, Result).
```

**Execution Order:**
1. Validate security (remote encryption required)
2. Execute pre-deploy hooks
3. Check for source changes
4. Backup current deployment
5. Generate and execute deployment script
6. Store new deployment hash
7. Execute post-deploy hooks

## Files Changed

- `src/unifyweaver/glue/deployment_glue.pl` - +521 lines (1,330 total)
- `tests/glue/test_deployment_glue.pl` - +190 lines (12 new tests)

## Tests

31 tests total (12 new for Phase 6b):

| Group | Tests |
|-------|-------|
| Multi-Host Support | 2 |
| Rollback Support | 4 |
| Health Check Integration | 1 |
| Hook Execution | 3 |
| Graceful Shutdown | 1 |
| Deploy with Hooks | 1 |

All tests passing.

## Test Plan

```bash
swipl -g "run_all_tests" -t halt tests/glue/test_deployment_glue.pl
```

## Phase 6 Progress

- [x] **Phase 6a**: Deployment foundation (SSH, lifecycle, security)
- [x] **Phase 6b**: Advanced deployment (this PR)
- [ ] **Phase 6c**: Error handling (retry, fallback, circuit breaker)
- [ ] **Phase 6d**: Monitoring (metrics, logging, alerts)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
