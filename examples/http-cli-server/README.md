# HTTP CLI Server Example

A web-based CLI interface for AI browser agents with JWT authentication, role-based access control, and WebSocket shell for superadmin users.

## Features

- **File Browser** - Navigate directories within sandbox
- **Search Commands** - grep, find, cat with glob expansion
- **Custom Commands** - Execute allowed shell commands
- **JWT Authentication** - Secure token-based auth
- **Role-Based Access Control** - user, admin, shell roles
- **WebSocket Shell** - Full interactive shell for superadmin
- **Mobile Support** - Text Mode and Capture Mode for touch devices

## Quick Start

```bash
# Install dependencies
npm install

# Start without auth (development)
npm start

# Start with auth enabled
npm run start:auth
```

Open http://localhost:3001 in your browser.

## Default Users

Created automatically on first login when auth is enabled:

| Email | Password | Roles | Access |
|-------|----------|-------|--------|
| shell@local | shell | shell, admin, user | Full access + WebSocket Shell |
| admin@local | admin | admin, user | All endpoints, no shell |
| user@local | user | user | Browse/search only |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_REQUIRED` | `false` | Enable JWT authentication |
| `PORT` | `3001` | Server port |
| `SANDBOX_ROOT` | `~/sandbox` | Root directory for file operations |
| `JWT_SECRET` | (dev secret) | JWT signing secret (change in production!) |
| `USERS_FILE` | `./users.json` | Path to users database |

## Endpoints

### Public (no auth required)
- `GET /auth/status` - Auth configuration
- `POST /auth/login` - Get JWT token
- `GET /auth/me` - Current user info

### Protected (all authenticated users)
- `GET /` - Web UI
- `POST /browse` - Browse directories
- `POST /grep` - Search file contents
- `POST /find` - Find files by pattern
- `POST /cat` - Read file contents
- `POST /feedback` - Submit/get feedback

### Admin Only (admin, shell roles)
- `POST /exec` - Execute custom commands

### WebSocket Shell (shell role only)
- `ws://localhost:3001/?token=JWT` - Interactive shell

## Shell Modes

The web UI provides two input modes for the shell:

- **Text Mode** (default) - Input field with Send button, works on mobile
- **Capture Mode** - Tap terminal to open keyboard, characters sent immediately

## Security Notes

- All file operations are sandboxed to `SANDBOX_ROOT`
- JWT tokens expire after 24 hours
- Passwords are hashed with SHA256 + salt
- WebSocket shell requires explicit "shell" role
- Change `JWT_SECRET` in production!

## Files

- `http-server.ts` - Main HTTP/WebSocket server
- `auth.ts` - JWT authentication module
- `command-proxy.ts` - Command validation and execution

## Future: UnifyWeaver Generation

This example is currently manually written. A future goal is to generate this server using UnifyWeaver's Prolog-based code generation system, similar to how Vue apps are generated from app specifications.
