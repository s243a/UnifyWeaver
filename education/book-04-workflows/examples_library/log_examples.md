# Log File Examples

This file contains various log file examples that can be referenced by workflows.

---

### Example: Simple Application Log

> [!example-record]
> id: 20251025-183500-001
> name: logs.app.simple_info

```text
2025-10-25 18:35:01 INFO: Application startup complete.
2025-10-25 18:35:02 INFO: User 'alice' logged in.
2025-10-25 18:35:05 WARNING: Configuration value 'timeout' is deprecated.
2025-10-25 18:35:10 INFO: User 'alice' accessed resource 'dashboard'.
```

---

### Example: Security Authentication Failure

> [!example-record]
> id: 20251025-183600-001
> name: logs.security.auth_failure

```text
2025-10-25 18:36:15 AUDIT: Login attempt for user 'admin' from IP 192.168.1.101.
2025-10-25 18:36:15 AUDIT: Authentication failed: incorrect password.
2025-10-25 18:36:20 AUDIT: Login attempt for user 'admin' from IP 192.168.1.101.
2025-10-25 18:36:20 AUDIT: Authentication failed: incorrect password.
2025-10-25 18:36:22 AUDIT: Brute-force attempt detected for user 'admin'. IP 192.168.1.101 temporarily blocked.
```
