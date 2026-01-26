# HTTP CLI Server Example

This example demonstrates generating an HTTP CLI server with authentication,
WebSocket shell, HTTPS support, and syntax-highlighted file viewing using
UnifyWeaver's declarative specification.

## Overview

Instead of manually writing server code, you define the server declaratively in
Prolog (`spec.pl`) and generate TypeScript code using UnifyWeaver's generators.

## Features

- **PTY Shell**: Real pseudo-terminal via node-pty with full terminal emulation
- **xterm.js Terminal**: Rich terminal UI with colors, cursor control, and proper keyboard handling
- **Text Mode Fallback**: Plain HTML fallback when xterm.js is unavailable or blocked
- **Syntax Highlighting**: Code viewing with highlight.js (20+ languages)
- **Root Selector**: Switch between sandbox, project, and home directories
- **Results Panel**: Download/copy buttons for grep, find, cat, exec output
- **File Upload**: Upload files with File System Access API support for Android 14+
- **File Download**: Download files from the server with proper MIME types
- **HTTPS Support**: Self-signed or custom certificates
- **Role-based Auth**: JWT tokens with shell/admin/user roles
- **Firewall Package Control**: Control which npm/CDN packages are allowed in generated code

## Files

| File | Description |
|------|-------------|
| `spec.pl` | Declarative server and auth specification |
| `generate.sh` | Script to run the generators |
| `generated/` | Output folder for generated TypeScript |

### Generators (in `src/unifyweaver/`)

| File | Description |
|------|-------------|
| `glue/http_server_generator.pl` | Server, routes, handlers, WebSocket PTY |
| `ui/html_interface_generator.pl` | HTML/CSS/Vue template generation |
| `ui/http_cli_ui.pl` | Declarative UI component specification |
| `ui/vue_generator.pl` | Vue.js component generation |

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
| `SANDBOX_ROOT` | Root directory for sandbox mode | `$HOME/sandbox` |
| `PROJECT_ROOT` | Root directory for project mode | `../../..` (project root) |
| `HOME_ROOT` | Root directory for home mode | `$HOME` |
| `JWT_SECRET` | Secret for JWT signing | `change-this-in-production` |
| `USERS_FILE` | Path to users database file | `users.txt` |

## Web Interface

The generated server includes a complete Vue.js single-page application:

### Tabs

| Tab | Description | Required Role |
|-----|-------------|---------------|
| Browse | File browser with navigation and download | user |
| Upload | Upload files to server | admin, shell |
| Grep | Regex search in files | user |
| Find | Find files by pattern | user |
| Cat | View file with syntax highlighting | user |
| Custom | Execute allowed commands | admin |
| Feedback | Submit feedback/notes | user |
| Shell | Interactive PTY terminal | shell |

### Root Selector

Switch between three directory roots:
- **Sandbox**: Isolated working directory (`$HOME/sandbox`)
- **Project**: UnifyWeaver project root
- **Home**: User's home directory

### Shell Features

- Real PTY via node-pty (not character emulation)
- **xterm.js** terminal emulation with catppuccin theme
- Auto-detects xterm.js availability on page load
- Falls back to text mode if xterm.js blocked or unavailable
- ANSI escape code stripping for text display
- Terminal resize support (via FitAddon)
- Three input modes:
  - **Terminal**: xterm.js with full keyboard handling
  - **Text Mode**: Simple input field for command entry
  - **Capture Mode**: Hidden input for mobile keyboards (fallback)

### Upload Features

- **File System Access API**: Uses `showOpenFilePicker()` on Chrome for better file browser
- **Android 14+ Support**: Works around Android 14's media-centric file picker limitations
- **Immediate Upload**: Files upload automatically after selection (no separate button)
- **Standard Fallback**: Falls back to `<input type="file">` on Firefox and older browsers
- **Size Limits**: 50MB per upload request
- **Path Validation**: Prevents directory traversal attacks

### Download Features

- **Browse Integration**: Download button appears when a file is selected
- **MIME Type Detection**: Proper Content-Type headers for common file types
- **Size Limits**: 100MB maximum download size

### Syntax Highlighting

File viewing uses highlight.js with support for:
JavaScript, TypeScript, Python, Ruby, Rust, Go, Java, C/C++,
HTML, CSS, JSON, YAML, Markdown, SQL, Prolog, and more.

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
  - `POST /upload` - Upload files (admin/shell only, multipart/form-data)
  - `GET /download` - Download files (authenticated)

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

## Firewall Package Control

The firewall system (`src/unifyweaver/core/firewall.pl`) can control which
packages are included in generated code:

```prolog
% Allow only specific npm packages
:- assertz(firewall:firewall_default([
    npm_packages([vue, xterm, '@xterm/addon-fit']),
    cdn_packages(['*unpkg.com*', '*jsdelivr.net*', '*cdnjs*'])
])).

% Deny xterm.js (forces text mode fallback)
:- assertz(firewall:firewall_default([
    denied([xterm])
])).
```

The generator checks `is_package_allowed/2` and uses fallback implementations
when packages are blocked by firewall policy.

## Related

- `../../src/unifyweaver/core/firewall.pl` - Package installation control
- `../../src/unifyweaver/sources/service_source.pl` - Service source type
- `../../src/unifyweaver/glue/http_server_generator.pl` - Server generator
- `../../src/unifyweaver/glue/auth_generator.pl` - Auth generator
- `../../docs/PROPOSAL_HTTP_CLI_GENERATION.md` - Design proposal
