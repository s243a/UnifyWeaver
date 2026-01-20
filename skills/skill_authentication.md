# Skill: Authentication

Declarative authentication backend configuration with multiple storage backends, password hashing algorithms, and token types.

## When to Use

- User asks "how do I add authentication to my app?"
- User needs user storage (file, database, LDAP)
- User wants password hashing (bcrypt, argon2, SHA-256)
- User needs JWT tokens or sessions

## Quick Start

```prolog
:- use_module('src/unifyweaver/glue/auth_backends').

% Configure auth for app
app(my_app, [
    auth([
        backend(text_file),
        password_hash(bcrypt),
        token_type(jwt),
        users_file('users.txt'),
        session_duration(86400)
    ])
]).

% Generate auth server
generate_auth_server(app(my_app, [...]), node, Files).
```

## Authentication Backends

### Available Backends

| Backend | Description | Status |
|---------|-------------|--------|
| `mock` | Client-side mock for development | Implemented |
| `text_file` | Simple text file storage | Implemented |
| `text_file_encrypted` | Encrypted text file | Proposed |
| `sqlite` | SQLite database | Proposed |
| `postgresql` | PostgreSQL database | Proposed |
| `mongodb` | MongoDB document store | Proposed |
| `ldap` | LDAP/Active Directory | Proposed |
| `oauth2` | OAuth2 providers (Google, GitHub, etc.) | Proposed |

### Backend Query

```prolog
% Check if backend is available (implemented)
backend_available(text_file).

% Get backend requirements
backend_requirements(postgresql, Reqs).
% Reqs = [package(postgresql), service(postgresql)]

% Get backend capabilities
backend_capabilities(mongodb, Caps).
% Caps = [persistence, document_store, scalable, flexible_schema]
```

### Backend Capabilities

| Backend | Capabilities |
|---------|--------------|
| `mock` | no_persistence, fast_setup, testing_only |
| `text_file` | persistence, simple_setup, portable, human_readable |
| `sqlite` | persistence, queryable, transactions, portable |
| `postgresql` | persistence, queryable, transactions, scalable, concurrent |
| `mongodb` | persistence, document_store, scalable, flexible_schema |
| `ldap` | external_auth, enterprise, centralized |
| `oauth2` | external_auth, social_login, delegated |

## Password Hashing

### Available Algorithms

| Algorithm | Security | Status |
|-----------|----------|--------|
| `plain` | None (development only) | Implemented |
| `bcrypt` | High | Implemented |
| `sha256` | Medium | Implemented |
| `argon2` | Very High | Proposed |
| `scrypt` | High | Proposed |

### Query Hash Algorithms

```prolog
% Check if algorithm is available
hash_available(bcrypt).

% Get algorithm properties
hash_algorithm(argon2, Props).
% Props includes: description, security, status
```

## Token Types

### Available Types

| Type | Description | Status |
|------|-------------|--------|
| `jwt` | JSON Web Token (stateless) | Implemented |
| `session` | Server-side session with cookie | Proposed |
| `api_key` | Static API key | Proposed |

### Token Properties

```prolog
token_type(jwt, Props).
% Props = [description(...), capabilities([stateless, expiry, claims]), status(implemented)]
```

## App Auth Configuration

### Extract Auth Config

```prolog
app_auth_config(AppSpec, AuthConfig).
app_auth_backend(AppSpec, Backend).
app_auth_hash(AppSpec, Hash).
app_auth_token_type(AppSpec, TokenType).
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `backend(Type)` | Storage backend | mock |
| `password_hash(Algo)` | Hashing algorithm | bcrypt |
| `token_type(Type)` | Token type | jwt |
| `users_file(Path)` | Users file path | users.txt |
| `session_duration(Sec)` | Token/session duration | 86400 |
| `encryption(Type)` | File encryption | none |

## Code Generation

### Generate Auth Config JSON

```prolog
generate_auth_config(AppSpec, ConfigJSON).
```

**Output:**
```json
{
  "backend": "text_file",
  "passwordHash": "bcrypt",
  "tokenType": "jwt",
  "usersFile": "users.txt",
  "sessionDuration": 86400
}
```

### Generate Auth Server

```prolog
generate_auth_server(AppSpec, Target, Files).
```

**Target:** `node` (generates Node.js/Express auth server)

**Generated Files:**
- `server/auth-config.json` - Configuration
- `server/auth-server.cjs` - Server implementation

### Generated Server Features

**Text File Backend:**
```javascript
// Functions available:
authenticate(username, password)  // Returns user or null
register(username, password, roles, permissions)
loadUsers()
hashPassword(password)
verifyPassword(password, storedHash)
```

**Mock Backend:**
```javascript
// Pre-configured test users:
// admin@test.com / admin123
// shell@test.com / shell123
// user@test.com / user123
```

## Users File Format

For `text_file` backend:

```
# Format: username:password_hash:roles:permissions
admin:$2b$12$...:admin,user:read,write,delete
user:$2b$12$...:user:read
guest:$2b$12$...:guest:read
```

### Generate Default Users

```prolog
generate_default_users(AppSpec, Content).
```

## Common Patterns

### Development Setup

```prolog
app(dev_app, [
    auth([
        backend(mock),
        password_hash(plain),
        token_type(jwt)
    ])
]).
```

### Production Setup

```prolog
app(prod_app, [
    auth([
        backend(text_file),
        password_hash(bcrypt),
        token_type(jwt),
        users_file('/secure/users.txt'),
        session_duration(3600)
    ])
]).
```

### Enterprise Setup

```prolog
app(enterprise_app, [
    auth([
        backend(ldap),
        token_type(jwt),
        ldap_server('ldap.company.com'),
        ldap_base_dn('dc=company,dc=com')
    ])
]).
```

### OAuth2 Setup

```prolog
app(social_app, [
    auth([
        backend(oauth2),
        providers([google, github]),
        token_type(jwt)
    ])
]).
```

## Related

**Parent Skill:**
- `skill_infrastructure.md` - Infrastructure sub-master

**Sibling Skills:**
- `skill_deployment.md` - Service deployment
- `skill_networking.md` - HTTP/socket generation
- `skill_frontend_security.md` - Frontend security

**Code:**
- `src/unifyweaver/glue/auth_backends.pl`
