# HTTP CLI Server Example

This example demonstrates generating an HTTP CLI server with authentication,
WebSocket shell, and HTTPS support using UnifyWeaver's declarative specification.

## Overview

Instead of manually writing server code, you define the server declaratively in
Prolog (`spec.pl`) and generate TypeScript code using UnifyWeaver's generators.

## Files

| File | Description |
|------|-------------|
| `spec.pl` | Declarative server and auth specification |
| `generate.sh` | Script to run the generators |
| `generated/` | Output folder for generated TypeScript |

## Usage

### Generate the server

```bash
./generate.sh
```

Or from the project root:

```bash
./examples/http-cli-server/generate.sh
```

### Run the generated server

```bash
cd generated
npm install
npx ts-node server.ts --port 8080
```

With HTTPS (generate self-signed cert first):

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"

# Run with HTTPS
npx ts-node server.ts --port 8080 --cert cert.pem --key key.pem
```

With authentication enabled:

```bash
AUTH_REQUIRED=true npx ts-node server.ts --port 8080 --cert cert.pem --key key.pem
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AUTH_REQUIRED` | Require login (`true`/`false`) | `false` |
| `SANDBOX_ROOT` | Root directory for file operations | `$HOME/sandbox` |
| `JWT_SECRET` | Secret for JWT signing | `change-this-in-production` |
| `USERS_FILE` | Path to users database file | `users.txt` |

## Specification

The `spec.pl` file defines:

### Service Specification

- **Port**: 3001
- **HTTPS**: Certificate and key from environment variables
- **Endpoints**:
  - `GET /health` - Health check (public)
  - `GET /auth/status` - Auth status (public)
  - `POST /auth/login` - Login (public)
  - `GET /auth/me` - Current user info (authenticated)
  - `GET /commands` - List available commands
  - `POST /browse` - Directory listing
  - `POST /grep` - Search file contents
  - `POST /find` - Find files by pattern
  - `POST /cat` - Read file contents
  - `POST /exec` - Execute commands (admin/shell only)
  - `POST /feedback` - Submit feedback

### Authentication Specification

- **Backend**: Text file (`users.txt`)
- **Password Hash**: bcrypt
- **Token Type**: JWT (24-hour expiry)
- **Roles**:
  - `shell` - Full access including WebSocket PTY
  - `admin` - Admin access with command execution
  - `user` - Basic browse and search only

### Default Users

| Email | Password | Roles |
|-------|----------|-------|
| shell@local | shell | shell, admin, user |
| admin@local | admin | admin, user |
| user@local | user | user |

## Comparison with Manual Implementation

The manual TypeScript implementation is preserved in `../prototypes/http-cli-server/`
for reference. The generated code should be functionally equivalent but is produced
from the declarative specification.

Benefits of the declarative approach:
- Single source of truth for configuration
- Easier to modify (change spec, regenerate)
- Can target multiple languages (TypeScript, Python, Go)
- Validates specification before generation

## Related

- `../../src/unifyweaver/sources/service_source.pl` - Service source type
- `../../src/unifyweaver/glue/http_server_generator.pl` - Server generator
- `../../src/unifyweaver/glue/auth_generator.pl` - Auth generator
- `../../docs/PROPOSAL_HTTP_CLI_GENERATION.md` - Design proposal
