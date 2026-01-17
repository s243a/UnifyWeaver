# feat(glue): Declarative shell sandbox and secure app generation

## Summary

Adds comprehensive security infrastructure for UnifyWeaver-generated applications, enabling declarative specification of:

- **Navigation guards** for route-level authentication/authorization
- **HTTPS support** for secure local development
- **Secure authentication backends** (FastAPI/Flask) with JWT and bcrypt
- **Shell sandbox system** with configurable security levels and multiple isolation backends

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

### HTTPS Support

```prolog
app(my_app, [
    https(true),
    ...
])
```

- Generates Vite config with SSL plugin
- Self-signed certificate handling for development

### Secure Authentication Backend

```prolog
app(my_app, [
    auth(secure, [backend(fastapi)]),
    ...
])
```

Generates complete auth backend with:
- JWT token authentication
- bcrypt password hashing
- User model with roles/permissions
- REST endpoints: register, login, logout, me

### Shell Sandbox System (`shell_sandbox.pl`)

Declarative security levels and sandbox backends:

```prolog
app(my_app, [
    shell_access([
        backends([proot, app_filter]),
        levels([
            level(trusted, [sandbox(app_filter), pty(true), preserve_home(false), ...]),
            level(sandbox, [sandbox(proot), commands([ls, cat, ...])]),
        ]),
        role_access([
            admin -> full,
            user -> trusted,
            guest -> sandbox
        ])
    ])
])
```

**Level options:**
- `sandbox(Backend)` - Isolation backend (none, app_filter, proot, etc.)
- `commands(List|all)` - Allowed commands
- `blocked_commands(List)` - Commands to block
- `pty(Bool)` - Enable PTY for interactive programs
- `preserve_home(Bool)` - Use real $HOME (true) or sandbox HOME (false)
- `timeout(Seconds)` - Command timeout

**Implemented backends:**
| Backend | Security | Description |
|---------|----------|-------------|
| `app_filter` | Low | Command whitelist/blocklist, pattern blocking |
| `proot` | Medium | Filesystem isolation via fake chroot (Termux compatible) |

**Proposed backends (documented):**
| Backend | Security | Status |
|---------|----------|--------|
| `firejail` | High | Proposed |
| `bubblewrap` | High | Proposed |
| `docker` | Very High | Proposed |
| `podman` | Very High | Proposed |
| `nsjail` | Very High | Proposed |

**Security levels:**
- `full` - No restrictions, PTY enabled
- `trusted` - Block destructive ops, PTY enabled
- `sandbox` - Isolated to ~/sandbox, limited commands
- `restricted` - Minimal read-only access
- `education` - Safe learning environment

## Files Changed

- `src/unifyweaver/glue/app_generator.pl` (+2150 lines)
  - Navigation guards for 4 frontend targets
  - HTTPS configuration generation
  - Secure auth backend generation (FastAPI/Flask)

- `src/unifyweaver/glue/shell_sandbox.pl` (+732 lines, new)
  - 9 sandbox backend definitions
  - 5 default security levels
  - Role-to-level mapping
  - JSON config generation

## Testing

Generated and tested a Vue app with:
- Navigation guards protecting routes by role
- HTTPS via Vite's basic-ssl plugin
- Web-based terminal with PTY support
- Configurable security levels (trusted, sandbox, etc.)
- proot filesystem isolation
- Interactive program support (tested with coro CLI)
- HOME directory isolation (tools like Claude Code get separate configs per shell context)

## Future Work

- [ ] Integrate `shell_sandbox.pl` into `app_generator.pl` for auto-generation
- [ ] Implement firejail/bubblewrap backends for Linux
- [ ] Add Docker backend for containerized isolation
- [ ] UI for security level selection in generated apps
