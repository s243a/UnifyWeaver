# Add Phase 6a: SSH Deployment and Lifecycle Management

## Summary

Implements Phase 6a "Deployment Foundation" of the cross-target glue system, providing SSH-based deployment, lifecycle management, change detection, and security validation.

## New Module: `deployment_glue.pl`

~820 lines of Prolog implementing automatic deployment infrastructure.

### Service Declarations
- `declare_service/2` - Register services with configuration (host, port, target, lifecycle)
- `declare_deploy_method/3` - Configure SSH or local deployment method
- `declare_service_sources/2` - Track source files for change detection
- `declare_lifecycle_hook/3` - Register pre/post deployment hooks

### Security by Default
- Remote services **require encryption** (HTTPS or SSH)
- HTTP only allowed for localhost
- `validate_security/2` - Check and report security violations
- `requires_encryption/1` / `is_local_service/1` - Query security requirements

### Change Detection
- `compute_source_hash/2` - SHA256 hash of all tracked source files
- `check_for_changes/2` - Returns `no_changes`, `changed(Old, New)`, or `never_deployed`
- `store_deployed_hash/2` - Record deployed version for future comparison

### Script Generation
- `generate_deploy_script/3` - Generate deployment script based on configured method
- `generate_ssh_deploy/3` - Full SSH deployment script with:
  - SSH agent check
  - Connectivity verification
  - rsync file synchronization
  - Dependency installation (pip, go mod, cargo)
  - Service restart with lifecycle hooks
- `generate_systemd_unit/3` - Generate systemd service unit files
- `generate_health_check_script/3` - Health check with configurable retries

### Lifecycle Management
- `deploy_service/2` - Execute deployment with change detection
- `start_service/2`, `stop_service/2`, `restart_service/2`
- Lifecycle hooks: `drain_connections`, `save_state`, `health_check`, `warm_cache`, `custom(Cmd)`

## Example Usage

```prolog
% Declare a Python ML service
:- declare_service(ml_predictor, [
    host('ml.example.com'),
    port(8080),
    target(python),
    entry_point('server.py'),
    transport(https)
]).

% Configure SSH deployment
:- declare_deploy_method(ml_predictor, ssh, [
    user('deploy'),
    remote_dir('/opt/services')
]).

% Track sources for change detection
:- declare_service_sources(ml_predictor, [
    'src/**/*.py',
    'models/*.pkl',
    'requirements.txt'
]).

% Add lifecycle hooks
:- declare_lifecycle_hook(ml_predictor, pre_shutdown, drain_connections).
:- declare_lifecycle_hook(ml_predictor, post_deploy, health_check).

% Generate deployment script
generate_deploy_script(ml_predictor, [], Script).
```

## Tests

19 test assertions across 9 test groups:

| Group | Tests |
|-------|-------|
| Service Declarations | 3 |
| Deployment Methods | 2 |
| Source Tracking | 2 |
| Security Validation | 5 |
| Lifecycle Hooks | 2 |
| SSH Deploy Script | 2 |
| Systemd Unit | 1 |
| Health Check Script | 1 |
| Local Deploy Script | 1 |

All tests passing.

## Files Changed

- `src/unifyweaver/glue/deployment_glue.pl` (new) - 820 lines
- `tests/glue/test_deployment_glue.pl` (new) - 350 lines

## Test Plan

```bash
swipl -g "run_all_tests" -t halt tests/glue/test_deployment_glue.pl
```

## Phase 6 Roadmap

- [x] **Phase 6a**: Deployment foundation (this PR)
- [ ] **Phase 6b**: Advanced deployment (rollback, multi-host, graceful shutdown)
- [ ] **Phase 6c**: Error handling (retry, fallback, circuit breaker)
- [ ] **Phase 6d**: Monitoring (health checks, metrics, logging)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
