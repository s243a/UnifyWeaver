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
npx ts-node server.ts
```

With HTTPS:

```bash
SSL_CERT=/path/to/cert.pem SSL_KEY=/path/to/key.pem npx ts-node server.ts
```

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
