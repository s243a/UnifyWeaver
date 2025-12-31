# Add Phase 6c: Error Handling - Retry, Fallback, Circuit Breaker

## Summary

Extends the deployment glue module with Phase 6c error handling features: retry policies with exponential backoff, fallback mechanisms, circuit breaker pattern, and timeout configuration.

## New Features

### Retry Policies

Configure automatic retry with exponential backoff for transient failures.

```prolog
% Configure retry policy
:- declare_retry_policy(ml_service, [
    max_retries(5),
    initial_delay(1000),      % 1 second
    max_delay(30000),         % Max 30 seconds
    backoff(exponential),     % exponential | linear | fixed
    multiplier(2),            % Delay multiplier
    retry_on([timeout, connection_refused, 503]),
    fail_on([400, 401, 403, 404])
]).

% Call with retry
call_with_retry(ml_service, predict, [Input], Result).
% Result = ok(Value) | error(max_retries_exceeded)
```

**Backoff Strategies:**
- `exponential`: Delay doubles each retry (1s, 2s, 4s, 8s...)
- `linear`: Delay increases by fixed amount
- `fixed`: Constant delay between retries

**New Predicates:**
- `declare_retry_policy/2` - Configure retry behavior
- `retry_policy/2` - Query retry configuration
- `call_with_retry/4` - Execute with automatic retries

### Fallback Mechanisms

Provide graceful degradation when primary operations fail.

```prolog
% Use backup service as fallback
:- declare_fallback(primary_service, backup_service(secondary_service)).

% Return default value on failure
:- declare_fallback(ml_service, default_value(fallback_prediction)).

% Use cached value
:- declare_fallback(data_service, cache([key(user_data), ttl(3600)])).

% Custom fallback predicate
:- declare_fallback(api_service, custom(my_fallback_handler)).

% Call with fallback
call_with_fallback(ml_service, predict, [Input], Result).
```

**Fallback Types:**
- `backup_service(Name)` - Try alternate service
- `default_value(Value)` - Return static default
- `cache(Options)` - Use cached response
- `custom(Predicate)` - Call custom handler

**New Predicates:**
- `declare_fallback/2` - Configure fallback behavior
- `fallback_config/2` - Query fallback configuration
- `call_with_fallback/4` - Execute with fallback on failure

### Circuit Breaker

Prevent cascade failures by tracking error rates and breaking circuits.

```prolog
% Configure circuit breaker
:- declare_circuit_breaker(ml_service, [
    failure_threshold(5),      % Open after 5 failures
    success_threshold(3),      % Close after 3 successes in half-open
    half_open_timeout(30000)   % Try half-open after 30 seconds
]).

% Call with circuit protection
call_with_circuit_breaker(ml_service, predict, [Input], Result).
% Result = ok(Value) | error(circuit_open) | error(Reason)

% Query circuit state
circuit_state(ml_service, State).
% State = closed | open | half_open

% Manually reset circuit
reset_circuit_breaker(ml_service).
```

**Circuit States:**
- `closed` - Normal operation, requests pass through
- `open` - Too many failures, requests rejected immediately
- `half_open` - Testing if service recovered

**New Predicates:**
- `declare_circuit_breaker/2` - Configure circuit breaker
- `circuit_breaker_config/2` - Query configuration
- `circuit_state/2` - Query current state
- `call_with_circuit_breaker/4` - Execute with circuit protection
- `reset_circuit_breaker/1` - Reset to closed state
- `record_circuit_failure/1` - Record failure (internal/testing)
- `record_circuit_success/1` - Record success (internal/testing)

### Timeout Configuration

Configure timeouts for service calls.

```prolog
% Configure timeouts
:- declare_timeouts(ml_service, [
    connect_timeout(5000),     % 5 second connection timeout
    read_timeout(30000),       % 30 second response timeout
    total_timeout(60000),      % 60 second total timeout
    idle_timeout(120000)       % 2 minute idle timeout
]).

% Call with timeout
call_with_timeout(ml_service, predict, [Input], Result).
% Result = ok(Value) | error(timeout)
```

**New Predicates:**
- `declare_timeouts/2` - Configure timeout values
- `timeout_config/2` - Query timeout configuration
- `call_with_timeout/4` - Execute with timeout

### Combined Protection

Apply all error handling strategies together.

```prolog
% Configure all protections
:- declare_retry_policy(ml_service, [max_retries(3), initial_delay(1000)]).
:- declare_fallback(ml_service, default_value(fallback)).
:- declare_circuit_breaker(ml_service, [failure_threshold(5)]).
:- declare_timeouts(ml_service, [total_timeout(30000)]).

% Single call applies all protections
protected_call(ml_service, predict, [Input], Result).
```

**Execution Order:**
1. Check circuit breaker (reject if open)
2. Apply timeout wrapper
3. Execute with retry policy
4. On failure, try fallback
5. Update circuit breaker state

## Files Changed

- `src/unifyweaver/glue/deployment_glue.pl` - +502 lines (1,872 total)
- `tests/glue/test_deployment_glue.pl` - +166 lines (15 new tests)

## Tests

46 tests total (15 new for Phase 6c):

| Group | Tests |
|-------|-------|
| Retry Policy | 4 |
| Fallback Mechanisms | 4 |
| Circuit Breaker | 4 |
| Timeout Configuration | 2 |
| Protected Call | 1 |

All tests passing.

## Test Plan

```bash
swipl -g "run_all_tests" -t halt tests/glue/test_deployment_glue.pl
```

## Phase 6 Progress

- [x] **Phase 6a**: Deployment foundation (SSH, lifecycle, security)
- [x] **Phase 6b**: Advanced deployment (rollback, multi-host, graceful shutdown)
- [x] **Phase 6c**: Error handling (this PR)
- [ ] **Phase 6d**: Monitoring (metrics, logging, alerts)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
