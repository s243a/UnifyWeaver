# HTTP CLI Server

A sandboxed HTTP interface for shell search commands, designed for AI browser agents (like Comet/Perplexity) to explore codebases.

## Overview

The HTTP CLI Server exposes `grep`, `find`, `cat`, and other read-only commands via HTTP endpoints, with a Vue-based web interface. It includes a feedback channel for agent-to-agent communication.

```bash
# Start server
SANDBOX_ROOT=/path/to/project npx ts-node src/unifyweaver/shell/http-server.ts

# Access at http://localhost:3001
```

## Target Audiences

### 1. AI Browser Agents (Primary)

**Examples:** Comet (Perplexity), Claude with web access, automated research tools

**Strengths:**
- Clean JSON API for programmatic access
- Structured feedback channel (`POST /feedback`) for observations
- Sandboxed execution prevents accidental modifications
- CORS enabled for browser-based agents

**Weaknesses:**
- Limited to read-only operations
- No command chaining or pipelines
- Cannot invoke build/test/compile workflows

**Best for:** Code exploration, searching for patterns, reading files, leaving observations for human review.

### 2. Shell-Literate Developers

**Examples:** Developers comfortable with grep, find, CLI flags

**Strengths:**
- Familiar command syntax - write commands as you would in terminal
- Options field accepts standard CLI flags (`-i`, `--include=*.ts`, `-type d`)
- Quick access without opening a terminal
- Safe sandbox prevents accidental `rm -rf` disasters

**Weaknesses:**
- Must know CLI flags (no GUI affordances)
- "Custom" tab's JSON array for arguments is unintuitive
- No tab completion or command history
- Limited command set (grep/find/cat/ls/head/tail/wc only)

**Best for:** Quick searches when you don't want to context-switch to terminal.

### 3. General Users / Non-Technical

**Strengths:**
- Web-based UI is accessible
- Tabbed interface separates operations clearly

**Weaknesses:**
- "Options (space-separated)" requires knowing CLI flags
- No guided controls (checkboxes for "case insensitive", dropdowns for file types)
- Error messages assume CLI familiarity
- No help text explaining what each option does

**Not recommended** for users unfamiliar with shell commands. A GUI with explicit controls would serve this audience better.

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | HTML/Vue interface |
| `/health` | GET | Server status and config |
| `/commands` | GET | List allowed commands |
| `/grep` | POST | Search file contents |
| `/find` | POST | Find files by pattern |
| `/cat` | POST | Read file contents |
| `/exec` | POST | Execute allowed command |
| `/feedback` | POST | Submit feedback (for agents) |
| `/feedback` | GET | Read feedback history |

### Example: Grep

```bash
curl -X POST http://localhost:3001/grep \
  -H "Content-Type: application/json" \
  -d '{"pattern": "export.*function", "path": "src/", "options": ["-i"]}'
```

### Example: Feedback

```bash
curl -X POST http://localhost:3001/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Found authentication logic in src/auth/",
    "type": "info",
    "context": "grep for validateToken"
  }'
```

## Design Decisions

### Why Shell-Style Input?

The tool targets users who already know shell commands. Rather than building GUI abstractions that hide the underlying operations, it exposes the actual command interface. This is:

- **Transparent:** You see exactly what command runs
- **Flexible:** Any valid flag works, not just predefined options
- **Educational:** Users learn real CLI skills

The tradeoff is accessibility - users must know flags like `-i` (case insensitive) or `-type d` (directories only).

### Why Read-Only?

The sandbox restricts operations to prevent:
- Accidental file deletion
- Writes outside the project
- Execution of arbitrary code

This is intentional for AI agent use cases where the agent should observe but not modify.

### Why Feedback Channel?

AI agents performing research need a way to record observations. The `/feedback` endpoint creates a structured log that:
- Timestamps each entry
- Categorizes by type (info, success, warning, error, suggestion)
- Preserves context for human review
- Enables agent-to-agent communication

## Limitations

1. **No command composition** - Cannot pipe `grep | xargs | sed`
2. **No UnifyWeaver integration** - Cannot invoke `uvw compile` or `uvw test`
3. **No persistent sessions** - Each request is stateless
4. **No write operations** - By design, but limits utility for some workflows
5. **GUI not beginner-friendly** - Requires CLI knowledge

## Future Directions

Potential improvements based on user feedback:

- **Explicit controls** - Checkboxes for common options (case-insensitive, file types)
- **List mode** - Dedicated `ls` operation with directories-only toggle
- **UnifyWeaver commands** - Expose compile/test as first-class operations
- **Command presets** - "Search TypeScript", "Find test files", etc.
- **Better Custom tab** - Simple text input instead of JSON array

## Files

- `src/unifyweaver/shell/http-server.ts` - Server implementation
- `src/unifyweaver/shell/command-proxy.ts` - Command validation
- `comet-feedback.log` - Feedback entries (in sandbox root)

## See Also

- [SHELL_COMMAND_CONSTRAINTS.md](./SHELL_COMMAND_CONSTRAINTS.md) - Constraint system
- [command-proxy.ts](../src/unifyweaver/shell/command-proxy.ts) - Allowed commands
