# Add Phase 6d: Monitoring - Health Checks, Metrics, Logging, Alerts

## Summary

Completes Phase 6 of the deployment glue module with monitoring capabilities: health check monitoring, Prometheus metrics export, structured JSON logging, and alerting with severity levels.

## New Features

### Health Check Monitoring

Configure and monitor service health with thresholds.

```prolog
% Configure health check
:- declare_health_check(api_service, [
    endpoint('/health'),
    interval(30),              % Check every 30 seconds
    timeout(5),                % 5 second timeout
    unhealthy_threshold(3),    % Mark unhealthy after 3 failures
    healthy_threshold(2)       % Mark healthy after 2 successes
]).

% Query health status
health_status(api_service, Status).
% Status = healthy | unhealthy | unknown

% Start monitoring
start_health_monitor(api_service, Result).
```

**New Predicates:**
- `declare_health_check/2` - Configure health check parameters
- `health_check_config/2` - Query configuration
- `health_status/2` - Query current status
- `start_health_monitor/2` - Start monitoring
- `stop_health_monitor/1` - Stop monitoring

### Metrics Collection

Collect and export metrics in Prometheus format.

```prolog
% Configure metrics
:- declare_metrics(api_service, [
    collect([request_count, latency, error_count]),
    labels([service-api_service, env-production]),
    export(prometheus),
    retention(3600)            % Keep metrics for 1 hour
]).

% Record metrics
record_metric(api_service, request_count, 1).
record_metric(api_service, latency, 150).

% Get all metrics
get_metrics(api_service, Metrics).
% Metrics = [metric(request_count, 1, Timestamp), ...]

% Export as Prometheus format
generate_prometheus_metrics(api_service, Output).
% Output = "api_service_request_count{service=\"api_service\"} 1\n..."
```

**Metric Types:**
- `counter(Name)` - Incrementing counter
- `gauge(Name)` - Point-in-time value
- `histogram(Name, Bucket)` - Distribution

**New Predicates:**
- `declare_metrics/2` - Configure metrics collection
- `metrics_config/2` - Query configuration
- `record_metric/3` - Record a metric value
- `get_metrics/2` - Get all metrics
- `generate_prometheus_metrics/2` - Export as Prometheus format

### Structured Logging

JSON and text formatted logging with level filtering.

```prolog
% Configure logging
:- declare_logging(api_service, [
    level(info),               % Minimum level: debug|info|warn|error
    format(json),              % json or text
    output(stdout),            % stdout or file(Path)
    max_entries(1000)          % Max entries to retain
]).

% Log events
log_event(api_service, info, 'Request received', [method-'GET', path-'/api']).
log_event(api_service, error, 'Database connection failed', [retry-3]).
```

**JSON Output:**
```json
{"timestamp":"2025-01-15T10:30:00Z","service":"api_service","level":"info","message":"Request received","method":"GET","path":"/api"}
```

**Text Output:**
```
[2025-01-15T10:30:00Z] info [api_service] Request received [method=GET path=/api]
```

**New Predicates:**
- `declare_logging/2` - Configure logging
- `logging_config/2` - Query configuration
- `log_event/4` - Log an event
- `get_log_entries/3` - Query log entries with filters

### Alerting

Define alerts with severity levels and notifications.

```prolog
% Define alerts
:- declare_alert(api_service, high_error_rate, [
    condition('error_rate > 0.05'),
    severity(critical),         % critical | warning | info
    cooldown(300),              % 5 minute cooldown between alerts
    notify([
        slack('#alerts'),
        email('oncall@example.com'),
        pagerduty
    ])
]).

:- declare_alert(api_service, high_latency, [
    condition('p99_latency > 1000'),
    severity(warning),
    notify([slack('#monitoring')])
]).

% Trigger an alert
trigger_alert(api_service, high_error_rate, [rate-0.1]).

% Check triggered alerts
check_alerts(api_service, TriggeredAlerts).
% TriggeredAlerts = [alert(high_error_rate, triggered, Since), ...]

% Get alert history
alert_history(api_service, [limit(100)], History).
```

**Notification Channels:**
- `slack(Channel)` - Slack channel
- `email(Address)` - Email notification
- `pagerduty` - PagerDuty integration
- `webhook(URL)` - Custom webhook

**New Predicates:**
- `declare_alert/3` - Define an alert
- `alert_config/3` - Query alert configuration
- `trigger_alert/3` - Trigger an alert
- `check_alerts/2` - Get triggered alerts
- `alert_history/3` - Query alert history

## Integration Example

```prolog
% Complete monitoring setup
:- declare_service(api_service, [
    host('api.example.com'),
    port(8080),
    transport(https)
]).

% Health monitoring
:- declare_health_check(api_service, [
    endpoint('/health'),
    interval(30)
]).

% Metrics
:- declare_metrics(api_service, [
    collect([requests, latency, errors]),
    export(prometheus)
]).

% Logging
:- declare_logging(api_service, [
    level(info),
    format(json)
]).

% Alerts
:- declare_alert(api_service, service_unhealthy, [
    severity(critical),
    notify([pagerduty])
]).

% Start monitoring
start_health_monitor(api_service, _).
```

## Files Changed

- `src/unifyweaver/glue/deployment_glue.pl` - +540 lines (2,460 total)
- `tests/glue/test_deployment_glue.pl` - +150 lines (16 new tests)

## Tests

62 tests total (16 new for Phase 6d):

| Group | Tests |
|-------|-------|
| Health Check Monitoring | 4 |
| Metrics Collection | 4 |
| Structured Logging | 4 |
| Alerting | 4 |

All tests passing.

## Test Plan

```bash
swipl tests/glue/test_deployment_glue.pl
```

## Phase 6 Complete

- [x] **Phase 6a**: Deployment foundation (SSH, lifecycle, security)
- [x] **Phase 6b**: Advanced deployment (rollback, multi-host, graceful shutdown)
- [x] **Phase 6c**: Error handling (retry, fallback, circuit breaker)
- [x] **Phase 6d**: Monitoring (this PR)

Phase 6 "Production Ready" is now complete. The deployment glue module provides:
- SSH-based deployment with agent forwarding
- Service lifecycle management
- Change detection and automatic redeployment
- Security validation (remote requires encryption)
- Multi-host deployment with rollback support
- Error handling with retry, fallback, and circuit breaker
- Health monitoring with alerts
- Prometheus metrics export
- Structured JSON logging

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
