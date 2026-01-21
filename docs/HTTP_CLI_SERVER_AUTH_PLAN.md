# HTTP CLI Server - Authentication Integration

## Overview

The HTTP CLI Server provides a web-based interface for AI browser agents (Comet/Perplexity) with JWT authentication, role-based access control, and a WebSocket shell for superadmin users.

## Implementation Status

- [x] Phase 1: Authentication
- [x] Phase 2: Role-Based Access
- [x] Phase 3: WebSocket Shell
- [ ] Phase 4: Security Hardening (future)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 HTTP CLI Server                      │
├─────────────────────────────────────────────────────┤
│  Public Endpoints:                                   │
│  ├── GET  /auth/status  - Auth configuration         │
│  ├── POST /auth/login   - Get JWT token              │
│  └── GET  /auth/me      - Current user info          │
├─────────────────────────────────────────────────────┤
│  Protected Endpoints (all roles):                    │
│  ├── GET  /             - Vue UI interface           │
│  ├── POST /browse       - File browser               │
│  ├── POST /grep         - Search file contents       │
│  ├── POST /find         - Find files by pattern      │
│  ├── POST /cat          - Read file contents         │
│  └── POST /feedback     - Submit/get feedback        │
├─────────────────────────────────────────────────────┤
│  Admin Endpoints (admin, shell roles):               │
│  └── POST /exec         - Execute custom commands    │
├─────────────────────────────────────────────────────┤
│  WebSocket Shell (shell role only):                  │
│  └── ws://localhost:3001/?token=JWT                  │
└─────────────────────────────────────────────────────┘
```

## Access Levels

| Role | Browse/Search | Custom Commands | WebSocket Shell |
|------|---------------|-----------------|-----------------|
| guest | No | No | No |
| user | Yes | No | No |
| admin | Yes | Yes | No |
| shell | Yes | Yes | Yes |

## Files

```
src/unifyweaver/shell/
├── http-server.ts      # Main server with auth + WebSocket
├── auth.ts             # JWT utilities, user storage
├── command-proxy.ts    # Command validation/execution
└── users.json          # Generated on first run
```

## Default Users

Created automatically on first login:

```
shell@local / shell  - roles: [shell, admin, user] - Full access
admin@local / admin  - roles: [admin, user]        - No shell
user@local  / user   - roles: [user]               - Browse only
```

## Configuration

Environment variables:

```bash
AUTH_REQUIRED=true    # Enable authentication (default: false)
SANDBOX_ROOT=...      # Sandbox directory (default: ~/sandbox)
JWT_SECRET=...        # JWT signing secret (change in production!)
USERS_FILE=...        # Path to users.json
```

## Usage

```bash
# Start with auth disabled (development)
npx ts-node src/unifyweaver/shell/http-server.ts

# Start with auth enabled
AUTH_REQUIRED=true npx ts-node src/unifyweaver/shell/http-server.ts
```

## API Examples

### Login

```bash
curl -X POST http://localhost:3001/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"shell@local","password":"shell"}'
```

Response:
```json
{
  "success": true,
  "data": {
    "token": "eyJhbG...",
    "user": {
      "id": "shell",
      "email": "shell@local",
      "roles": ["shell", "admin", "user"],
      "permissions": ["read", "write", "delete", "shell"]
    }
  }
}
```

### Protected Endpoint

```bash
curl -X POST http://localhost:3001/browse \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"path":"."}'
```

### WebSocket Shell

```javascript
const ws = new WebSocket('ws://localhost:3001/?token=' + token);
ws.send(JSON.stringify({ type: 'input', data: 'ls\r' }));
```

## Vue UI Features

1. **Login Screen** - Shown when auth required and not logged in
2. **User Header** - Shows email, roles, and logout button
3. **Tabs** - Browse, Grep, Find, Cat, Custom, Feedback
4. **Shell Tab** - Only visible to shell role users
5. **Real-time Shell** - WebSocket-based interactive shell

## Security Notes

- JWT tokens expire after 24 hours
- Passwords are hashed with SHA256 + salt
- Shell commands run in sandbox directory
- WebSocket shell requires explicit "shell" role
- All protected endpoints require valid JWT when AUTH_REQUIRED=true

## Future Improvements (Phase 4)

- [ ] Rate limiting on login attempts
- [ ] Token refresh mechanism
- [ ] Token revocation (logout)
- [ ] HTTPS support
- [ ] Audit logging
