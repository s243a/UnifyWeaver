# Prototypes

Experimental tools and proof-of-concept implementations.

## HTTP CLI Server

A sandboxed HTTP interface for shell search commands, designed for AI browser agents.

**Location:** `src/unifyweaver/shell/http-server.ts`

**Features:**
- File browser with working directory support
- Shell commands: cd, pwd, grep, find, cat, head, tail, ls, wc
- Glob expansion for patterns like `*.ts`, `*/`
- Feedback channel for agent communication
- Vue 3 web interface

**Usage:**
```bash
SANDBOX_ROOT=/path/to/project npx ts-node src/unifyweaver/shell/http-server.ts
# Access at http://localhost:3001
```

**Documentation:** See [docs/HTTP_CLI_SERVER.md](../../docs/HTTP_CLI_SERVER.md)

**Status:** Prototype - works but needs security hardening (authentication, HTTPS) before production use.
