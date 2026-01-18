# Skill: Webapp Security

Configure authentication, authorization, TLS, and shell sandboxing for generated web applications.

## When to Use

- User wants to add authentication to a generated app
- User asks "how do I secure my app?"
- User needs navigation guards or route protection
- User wants to configure TLS/HTTPS
- User needs shell command sandboxing

## Quick Start

```prolog
?- use_module('src/unifyweaver/glue/auth_backends').
?- generate_complete_project(
       app(myapp, [
           auth([
               backend(jwt),
               guards([authenticated])
           ]),
           navigation(tabs, [...], [])
       ]),
       [frontend-vue, backend-fastapi],
       '/tmp/secure_app',
       Result
   ).
```

## Commands

### Add JWT Authentication
```prolog
?- generate_complete_project(
       app(myapp, [
           auth([
               backend(jwt),
               secret_env('JWT_SECRET'),
               expiry(3600)
           ]),
           navigation(...)
       ]),
       [frontend-vue, backend-fastapi],
       'OUTPUT_DIR',
       Result
   ).
```

### Add Navigation Guards
```prolog
?- generate_complete_project(
       app(myapp, [
           auth([
               guards([
                   guard(authenticated, '/login'),
                   guard(admin, '/unauthorized')
               ])
           ]),
           navigation(tabs, [
               screen(home, 'HomeView', []),
               screen(admin, 'AdminView', [guard(admin)])
           ], [])
       ]),
       [frontend-vue],
       'OUTPUT_DIR',
       Result
   ).
```

### Configure TLS
```prolog
?- use_module('src/unifyweaver/glue/tls_config').
?- generate_tls_config(
       tls_options([
           cert_file('/path/to/cert.pem'),
           key_file('/path/to/key.pem'),
           min_version('TLSv1.2')
       ]),
       ConfigOutput
   ).
```

### Shell Sandbox for Backend
```prolog
?- use_module('src/unifyweaver/glue/shell_sandbox').
?- sandbox_command(
       'ls -la /tmp',
       [
           allowed_paths(['/tmp', '/var/data']),
           denied_commands(['rm', 'sudo']),
           timeout(30)
       ],
       Result
   ).
```

## Authentication Backends

| Backend | Use Case | Options |
|---------|----------|---------|
| `jwt` | Stateless API auth | `secret_env`, `expiry`, `algorithm` |
| `session` | Server-side sessions | `store`, `cookie_name` |
| `oauth2` | Third-party login | `provider`, `client_id`, `redirect_uri` |
| `basic` | Simple auth | `realm` |

## Guard Types

| Guard | Description |
|-------|-------------|
| `authenticated` | User must be logged in |
| `admin` | User must have admin role |
| `role(RoleName)` | User must have specific role |
| `permission(Perm)` | User must have specific permission |

## Shell Sandbox Options

| Option | Description |
|--------|-------------|
| `allowed_paths([...])` | Whitelist of filesystem paths |
| `denied_commands([...])` | Blacklist of dangerous commands |
| `timeout(Seconds)` | Maximum execution time |
| `max_output(Bytes)` | Maximum output size |

## Related

**Skills:**
- `skill_app_generation.md` - Generate the base application
- `skill_unifyweaver_compile.md` - General compilation

**Documentation:**
- `docs/FIREWALL_GUIDE.md` - Network security policies
- `docs/CONTROL_PLANE.md` - Service restrictions

**Education (in `education/` subfolder):**
- `book-08-security-firewall/01_introduction.md` - Security overview
- `book-08-security-firewall/02_firewall_policies.md` - Network policies
- `book-08-security-firewall/03_lifecycle_hooks.md` - Security hooks
- `book-08-security-firewall/04_target_security.md` - Target-specific security
- `book-08-security-firewall/05_validation_fallback.md` - Input validation
- `book-08-security-firewall/06_production_hardening.md` - Production security

**Code:**
- `src/unifyweaver/glue/auth_backends.pl` - Authentication backends
- `src/unifyweaver/glue/tls_config.pl` - TLS configuration
- `src/unifyweaver/glue/shell_sandbox.pl` - Shell sandboxing
- `src/unifyweaver/glue/rpyc_security.pl` - RPyC security
