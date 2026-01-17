# feat(glue): Declarative security infrastructure for app generation

## Summary

Adds comprehensive security infrastructure for UnifyWeaver-generated applications, enabling declarative specification of:

- **Navigation guards** for route-level authentication/authorization
- **Authentication backends** with configurable storage and hashing
- **TLS/SSL configuration** with multiple termination strategies
- **Shell sandbox system** with configurable security levels and isolation backends

## Features

### Navigation Guards (`app_generator.pl`)

Route-level security for Vue, React Native, Flutter, and SwiftUI:

```prolog
navigation(router, [
    screen(home, 'HomeView', []),
    screen(dashboard, 'DashboardView', [guards([auth])]),
    screen(admin, 'AdminPanel', [guards([auth, admin])]),
    screen(profile, 'ProfileView', [protected(true)])
], [])
```

- Composable guards: `guards([auth])`, `guards([auth, admin])`
- Shorthand: `protected(true)` expands to `guards([auth])`
- Auto-generates auth stores, login screens, and unauthorized views

### Authentication Backends (`auth_backends.pl`)

Declarative authentication storage configuration:

```prolog
app(my_app, [
    auth([
        backend(text_file),        % Default: mock
        password_hash(bcrypt),     % Default: bcrypt
        token_type(jwt),           % Default: jwt
        users_file('users.txt'),   % Default: users.txt
        session_duration(86400)    % Default: 86400 (24h)
    ])
])
```

**Storage backends:**
| Backend | Description | Status |
|---------|-------------|--------|
| `mock` (default) | Client-side mock for development | Implemented |
| `text_file` | Text file with hashed passwords | Implemented |
| `text_file_encrypted` | Encrypted text file | Proposed |
| `sqlite` | SQLite database | Proposed |
| `postgresql` | PostgreSQL database | Proposed |
| `mongodb` | MongoDB document store | Proposed |
| `ldap` | LDAP/Active Directory | Proposed |
| `oauth2` | OAuth2 providers | Proposed |

**Password hashing:**
| Algorithm | Security | Status |
|-----------|----------|--------|
| `bcrypt` (default) | High | Implemented |
| `sha256` | Medium | Implemented |
| `plain` | None (dev only) | Implemented |
| `argon2` | Very High | Proposed |
| `scrypt` | High | Proposed |

### TLS/SSL Configuration (`tls_config.pl`)

Declarative TLS termination strategies:

```prolog
app(my_app, [
    tls([
        mode(proxy),                  % Default: proxy
        cert_source(auto_generated),  % Default: auto_generated
        cert_path('certs/server.crt'),
        key_path('certs/server.key'),
        https_port(443)               % Default: 443
    ])
])
```

**TLS modes:**
| Mode | Description | Use Case |
|------|-------------|----------|
| `proxy` (default) | Dev proxy terminates TLS, backends plain HTTP | Development |
| `shared_cert` | All servers share same certificate files | Production |
| `reverse_proxy` | External nginx/caddy handles TLS | Production |
| `per_server` | Each server manages its own certificate | Development |

**Certificate sources:**
| Source | Description | Status |
|--------|-------------|--------|
| `auto_generated` (default) | Self-signed certificate | Implemented |
| `file` | Load from cert/key files | Implemented |
| `env` | Load from environment variables | Implemented |
| `lets_encrypt` | ACME protocol | Proposed |
| `vault` | HashiCorp Vault | Proposed |

**Config generators:**
- `generate_vite_proxy_config/2` - Vite dev server proxy
- `generate_nginx_config/3` - Nginx reverse proxy
- `generate_caddy_config/3` - Caddy reverse proxy
- `generate_shared_cert_loader/2` - Node.js shared cert loader

### Shell Sandbox System (`shell_sandbox.pl`)

Declarative security levels and sandbox backends:

```prolog
app(my_app, [
    shell_access([
        backends([proot, app_filter]),
        levels([
            level(superadmin, [sandbox(none), pty(true), preserve_home(true), auto_shell(true)]),
            level(trusted, [sandbox(app_filter), pty(true), preserve_home(false)]),
            level(sandbox, [sandbox(proot), commands([ls, cat, ...])]),
        ]),
        role_access([
            shell -> superadmin,
            admin -> full,
            user -> trusted,
            guest -> sandbox
        ])
    ])
])
```

**Level options:**
| Option | Description | Default |
|--------|-------------|---------|
| `sandbox(Backend)` | Isolation backend | `app_filter` |
| `commands(List\|all)` | Allowed commands | `all` |
| `blocked_commands(List)` | Commands to block | `[]` |
| `pty(Bool)` | Enable PTY for interactive programs | `false` |
| `preserve_home(Bool)` | Use real $HOME or sandbox HOME | `false` |
| `auto_shell(Bool)` | Auto-start PTY on connect | `false` |
| `timeout(Seconds)` | Command timeout | `120` |

**Default security levels:**
| Level | Sandbox | PTY | preserve_home | auto_shell |
|-------|---------|-----|---------------|------------|
| `superadmin` | none | true | true | true |
| `full` | none | true | false | false |
| `trusted` | app_filter | true | false | false |
| `sandbox` | proot | false | false | false |
| `restricted` | proot | false | false | false |
| `education` | proot | false | false | false |

**Implemented backends:**
| Backend | Security | Description |
|---------|----------|-------------|
| `app_filter` | Low | Command whitelist/blocklist, pattern blocking |
| `proot` | Medium | Filesystem isolation via fake chroot (Termux compatible) |

**Proposed backends:**
| Backend | Security | Status |
|---------|----------|--------|
| `firejail` | High | Proposed |
| `bubblewrap` | High | Proposed |
| `docker` | Very High | Proposed |
| `podman` | Very High | Proposed |
| `nsjail` | Very High | Proposed |

## Files Changed

- `src/unifyweaver/glue/app_generator.pl` (+2150 lines)
  - Navigation guards for 4 frontend targets
  - HTTPS configuration generation

- `src/unifyweaver/glue/shell_sandbox.pl` (+800 lines, new)
  - 9 sandbox backend definitions
  - 6 default security levels (including superadmin)
  - Role-to-level mapping
  - JSON config generation

- `src/unifyweaver/glue/auth_backends.pl` (+503 lines, new)
  - 8 auth backend definitions
  - 5 password hashing algorithms
  - 3 token types
  - Server code generation

- `src/unifyweaver/glue/tls_config.pl` (+558 lines, new)
  - 4 TLS modes
  - 5 certificate sources
  - Nginx/Caddy config generators
  - Shared certificate loader

## Testing

Generated and tested a Vue app with:
- Navigation guards protecting routes by role
- Real auth backend with bcrypt + JWT (text file storage)
- HTTPS via Vite proxy (single certificate)
- Web-based terminal with PTY support
- Superadmin role with auto-shell and real HOME
- Configurable security levels (trusted, sandbox, etc.)
- proot filesystem isolation
- Interactive program support (tested with Claude Code, coro CLI)

## Future Work

- [ ] Integrate modules into `app_generator.pl` for auto-generation
- [ ] Implement SQLite/PostgreSQL auth backends
- [ ] Implement firejail/bubblewrap sandbox backends
- [ ] Add Let's Encrypt certificate automation
- [ ] UI for security level selection in generated apps
