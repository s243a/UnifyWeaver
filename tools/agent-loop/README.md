# UnifyWeaver Agent Loop

A terminal-friendly AI assistant that works reliably in Termux and constrained terminal environments.

## Why Agent Loop?

Interactive TUI-based coding agents (like coro-code, aider) often break in Termux and limited terminals due to escape code / cursor control issues. This tool bypasses those problems by:

1. Calling AI backends in **single-task mode** (no fancy TUI)
2. Managing conversation context ourselves
3. Parsing and executing tool calls
4. Providing clean, simple text output

## Quick Start

```bash
cd tools/agent-loop/generated

# Interactive mode (default: coro backend)
python3 agent_loop.py

# Single prompt
python3 agent_loop.py "list files in current directory"

# Use different backends
python3 agent_loop.py -b claude "prompt"        # Claude API
python3 agent_loop.py -b claude-code "prompt"   # Claude Code CLI
python3 agent_loop.py -b openai "prompt"        # OpenAI API
python3 agent_loop.py -b gemini "prompt"        # Gemini CLI
python3 agent_loop.py -b ollama-api "prompt"    # Ollama REST API
python3 agent_loop.py -b ollama-cli "prompt"    # Ollama CLI
```

## Installation

### Requirements

- Python 3.8+
- One of the supported backends installed

### Optional Dependencies

```bash
# For Claude API backend
pip install anthropic

# For OpenAI API backend
pip install openai

# For YAML config files
pip install pyyaml

# For rich text rendering (optional)
pip install rich
```

## Backends

| Backend | Command | Requirements |
|---------|---------|--------------|
| `coro` | `coro` | coro CLI installed |
| `claude` | Claude API | `ANTHROPIC_API_KEY` env var |
| `claude-code` | `claude` | claude-code CLI installed |
| `openai` | OpenAI API | `OPENAI_API_KEY` env var |
| `gemini` | `gemini` | gemini CLI installed |
| `openrouter` | OpenRouter API | `OPENROUTER_API_KEY` or `uwsal.json` / `coro.json` |
| `ollama-api` | Ollama REST | Ollama server running |
| `ollama-cli` | `ollama run` | ollama CLI installed |

## Configuration

### Agent Variants

Create a config file to define reusable agent configurations:

```bash
# Create example config
python3 agent_loop.py --init-config agents.yaml

# List available agents
python3 agent_loop.py --list-agents

# Use an agent variant
python3 agent_loop.py -a yolo "create hello.py"
python3 agent_loop.py -a claude-opus "complex task"
```

Example `agents.yaml`:

```yaml
agents:
  yolo:
    description: "Fast, auto-approve tools"
    backend: coro
    auto_tools: true
    max_iterations: 0

  claude-opus:
    description: "Claude Opus via API"
    backend: claude
    model: claude-opus-4-20250514
    stream: true

  ollama:
    description: "Local Ollama"
    backend: ollama-api
    model: codellama
    host: localhost
    port: 11434
```

## Interactive Commands

When running interactively, use these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/backend <name>` | Switch backend |
| `/iterations <n>` | Set max tool iterations |
| `/format <fmt>` | Set context format (plain/markdown/json/xml) |
| `/save [name]` | Save current session |
| `/load <id>` | Load a saved session |
| `/list` | List saved sessions |
| `/export <file>` | Export to markdown/HTML/JSON/text |
| `/cost` | Show API cost summary |
| `/search <query>` | Search saved sessions |
| `/alias` | Manage command aliases |
| `/template` | Manage prompt templates |
| `/history` | View/manage conversation history |
| `/clear` | Clear conversation history |
| `/quit` or `/exit` | Exit |

## Features

### Tool Handling

The agent loop parses and executes tool calls from AI responses:

- **Bash** - Execute shell commands (with confirmation)
- **Read** - Read file contents
- **Write** - Write files (with confirmation)
- **Edit** - Edit files using search/replace

Use `--auto-tools` to skip confirmations, or `--no-tools` to disable execution.

#### Security

Tool execution uses a layered security model: path validation, command blocklist, command proxying, and audit logging. The security posture is controlled via profiles.

```bash
# Default: cautious (path validation + command blocklist)
python3 agent_loop.py -b openrouter --auto-tools "Read /etc/shadow"
# → [Security] Blocked: /etc/shadow is a sensitive system file

# Disable all security checks
python3 agent_loop.py --no-security "prompt"

# Use a specific profile
python3 agent_loop.py --security-profile paranoid "prompt"
```

| Profile | Path validation | Command blocklist | Command proxy | Audit | Confirmation |
|---------|----------------|-------------------|---------------|-------|--------------|
| `open` | Off | Off | Off | Off | Normal |
| `cautious` | On | On (default) | Off | Basic | Normal |
| `sandboxed` | On | On + extra blocks | Enabled | Detailed | Normal |
| `paranoid` | On | Allowlist-only | Strict | Forensic | Safe commands skip; others prompt |

**Profile details:**

- **`open`** — No restrictions. For trusted manual use.
- **`cautious`** — Default. Blocks dangerous paths (e.g. `~/.ssh/`, `/etc/shadow`) and commands (e.g. `rm -rf /`, `curl | bash`). Basic audit logging.
- **`sandboxed`** — Extra command blocks (`sudo`, `eval`, `nohup`, backgrounding, inline `os.system`/`subprocess`). Command proxy validates rm, curl, wget, python, git, ssh before execution. Network restricted to localhost. Detailed audit logging.
- **`paranoid`** — Allowlist-only mode: only explicitly permitted commands can run (ls, cat, grep, git status, find, python3 *.py, node *.js, etc.). Safe read-only commands (ls, cat, grep, etc.) run without confirmation. Potentially dangerous commands (find, python3, node) still prompt. Strict proxy blocks force push, hard reset, pipe-to-shell, etc. Forensic audit logging. File size limits (1 MB read, 10 MB write).

**Command proxy** (sandboxed/paranoid): validates commands in-process before `subprocess.run()`:

| Command | Rules |
|---------|-------|
| `rm` | Blocks `rm -rf /`, `/home`, `~`, `/etc`, `/usr` |
| `curl`/`wget` | Blocks pipe-to-shell, writes to `/etc/` |
| `python`/`python3` | Blocks `-c` with `os.system`, `subprocess`, `eval`, `exec` |
| `git` | Blocks `reset --hard`, `clean -f`, `push --force`; warns on push/pull/merge |
| `ssh` | Blocks `ProxyCommand`; fully blocked in strict mode |
| `scp`/`nc`/`netcat` | Fully blocked in strict mode |

**Audit logging** writes JSONL to `~/.agent-loop/audit/` (configurable via `uwsal.json`). Levels: `basic` (commands + tools), `detailed` (+ file access, API calls), `forensic` (+ output, timing).

Blocklists are customizable via `uwsal.json`:

```json
{
  "security": {
    "profile": "cautious",
    "blocked_paths": ["/data/production/"],
    "allowed_paths": ["/etc/hosts"],
    "blocked_commands": ["\\bdrop\\s+database\\b"],
    "allowed_commands": ["rm -rf ./build"],
    "audit_log_dir": "~/.agent-loop/audit/"
  }
}
```

### Session Management

```bash
# Save current session
/save my-project

# List saved sessions
python3 agent_loop.py --list-sessions

# Load a session
python3 agent_loop.py -s abc123

# Search across sessions
python3 agent_loop.py --search "authentication"
```

### Streaming

Enable streaming for supported backends:

```bash
python3 agent_loop.py -b claude --stream "explain this code"
```

### Cost Tracking

API costs are tracked automatically for Claude and OpenAI:

```bash
# View costs
/cost

# Disable tracking
python3 agent_loop.py --no-cost-tracking "prompt"
```

### Export

Export conversations to various formats:

```bash
/export conversation.md    # Markdown
/export chat.html          # HTML
/export data.json          # JSON
/export raw.txt            # Plain text
```

### Shell Completions

Install shell completions for bash or zsh:

```bash
# Bash
source tools/agent-loop/generated/completions.bash

# Zsh
source tools/agent-loop/generated/completions.zsh
```

### Multi-line Input

Enter multi-line prompts using:

- **Code blocks**: Start with \`\`\` and end with \`\`\`
- **Heredoc**: Start with `<<EOF` and end with `EOF`
- **Continuation**: End line with `\`

### Prompt Templates

Save and reuse prompt templates with variable substitution:

```bash
/template add review "Review this code for {focus}: {code}"
/template use review focus=security code="$(cat main.py)"
```

### Command Aliases

Create shortcuts for common operations:

```bash
/alias add q quit
/alias add yolo "backend coro; auto-tools"
```

## Command Resolution

CLI backends (`coro`, `claude-code`, `gemini`, `ollama-cli`) automatically resolve which command to use:

1. If `--command` is specified, use that exactly (fail if not found)
2. Try the default command for the backend
3. Try fallback commands (e.g., `coro` falls back to `claude`)
4. Print a warning when using a fallback

```bash
# Disable fallback behavior
python3 agent_loop.py -b coro --no-fallback "prompt"

# Specify exact command
python3 agent_loop.py -b coro --command /usr/local/bin/coro "prompt"
```

### API Key and Config Resolution

API backends resolve credentials through a priority chain:

```
1. --api-key CLI argument
2. Backend-specific env var (OPENROUTER_API_KEY, ANTHROPIC_API_KEY, etc.)
3. uwsal.json (keys.<backend> or top-level api_key)
4. coro.json fallback
5. Standard file locations (~/.anthropic/api_key, etc.)
```

Config files are searched in order: `CWD/uwsal.json` → `~/uwsal.json` → `CWD/coro.json` → `~/coro.json`.

Use `--no-fallback` to skip `coro.json` (uwsal.json is always checked).

## Context Modes

| Mode | Description |
|------|-------------|
| `continue` | Keep full conversation history |
| `fresh` | Start fresh each turn |
| `sliding` | Keep recent messages only |

```bash
python3 agent_loop.py --context-mode sliding "prompt"
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | API key for Claude backend |
| `OPENAI_API_KEY` | API key for OpenAI backend |
| `OPENROUTER_API_KEY` | API key for OpenRouter backend |
| `GEMINI_API_KEY` | API key for Gemini backend |
| `AGENT_LOOP_CONFIG` | Default config file path |
| `AGENT_LOOP_SESSIONS` | Sessions directory |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    UnifyWeaver Agent Loop                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Input     │    │   Agent     │    │   Output    │      │
│  │   Handler   │───▶│   Router    │───▶│   Handler   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         │                 │                   │              │
│         ▼                 ▼                   ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Context   │    │  Tool Call  │    │   Display   │      │
│  │   Manager   │◀──▶│   Handler   │    │   Renderer  │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      Agent Backends                          │
├─────────┬─────────┬─────────┬─────────┬─────────┬───────────┤
│  Coro   │ Claude  │ OpenAI  │ Ollama  │ Gemini  │ Claude    │
│  (CLI)  │  (API)  │  (API)  │ (API)   │  (CLI)  │ Code(CLI) │
└─────────┴─────────┴─────────┴─────────┴─────────┴───────────┘
```

## Module Reference

| Module | Description |
|--------|-------------|
| `agent_loop.py` | Main entry point and REPL |
| `backends/` | Backend implementations |
| `config.py` | Configuration and agent variants |
| `context.py` | Conversation context management |
| `tools.py` | Tool call parsing and execution |
| `security/` | Security subsystem (audit, profiles, proxy) |
| `security/audit.py` | JSONL audit logging |
| `security/profiles.py` | Security profile definitions |
| `security/proxy.py` | In-process command proxy |
| `sessions.py` | Session save/load |
| `export.py` | Export to various formats |
| `costs.py` | API cost tracking |
| `search.py` | Session search |
| `retry.py` | Retry logic with backoff |
| `aliases.py` | Command aliases |
| `templates.py` | Prompt templates |
| `history.py` | History management |
| `multiline.py` | Multi-line input handling |
| `display.py` | Spinner, progress bar, terminal control (tput) |
| `skills.py` | Skills and agent.md loading |

## Examples

### Basic Usage

```bash
# Ask a question
python3 agent_loop.py "What is the capital of France?"

# Code task
python3 agent_loop.py "Create a Python script that counts words in a file"
```

### With Configuration

```bash
# Use configured agent
python3 agent_loop.py -a claude-opus "Explain this error: $ERROR"

# Override model
python3 agent_loop.py -b claude -m claude-sonnet-4-20250514 "prompt"
```

### Session Workflow

```bash
# Start interactive session
python3 agent_loop.py

> Help me build a REST API
[conversation...]

> /save api-project

# Later, resume
python3 agent_loop.py --list-sessions
python3 agent_loop.py -s <session-id>
```

## Testing Status

### Tested Features

| Feature | Backend | Status |
|---------|---------|--------|
| Single prompt mode | claude-code, coro, gemini | Working |
| Interactive REPL | claude-code, coro | Working |
| Config generation (`--init-config`) | - | Working |
| Session listing (`--list-sessions`) | - | Working |
| Agent variants (`-a`) | claude-code | Working |
| Context modes (continue, fresh, sliding) | claude-code, coro | Working |
| Help output | - | Working |
| Gemini CLI backend (`-b gemini`) | gemini | Working |
| Fancy mode (`--fancy`) spinner | coro, claude-code, gemini | Working |
| Stream-json live tool progress | claude-code, gemini | Working |
| Context limits (`--max-chars`, `--max-words`, `--max-tokens`) | openrouter, coro | Working |
| Token estimation (chars/4 heuristic) | all | Working |
| Sliding window uses `max_messages` | coro | Verified |
| Duplicate message fix (API backends) | claude-api, openai-api | Verified |
| `on_status` via `**kwargs` (no try/except) | all | Verified |
| Command resolution with fallbacks | coro, claude-code, gemini | Working |
| `--no-fallback` flag | coro | Working |
| Coro debug mode + token parsing | coro | Working |
| Coro `--max-steps` default (5) | coro | Working |
| Coro config discovery (`~/coro.json`) | coro | Working |
| ANSI escape stripping in coro output | coro | Working |
| Spinner elapsed time display | all (fancy) | Working |
| OpenRouter API backend | openrouter | Working |
| OpenRouter auto-config from uwsal.json/coro.json | openrouter | Working |
| OpenRouter pricing auto-fetch | openrouter, coro | Working |
| Context limits with OpenRouter | openrouter | Verified |
| Context limits with Coro | coro | Verified |
| Coro `max_token` passthrough (`--max-tokens`) | coro | Working |
| Fix: `context or ContextManager()` truthiness bug | all | Fixed |
| OpenRouter SSE streaming | openrouter | Working |
| Paste detection (timing-based) | all | Working |
| Unified config (uwsal.json + coro.json cascade) | openrouter, coro | Working |
| Per-provider API keys (`keys` object in uwsal.json) | openrouter | Working |
| `--no-fallback` skips coro.json for config | openrouter, coro | Working |
| Security: path validation + command blocklist | openrouter | Working |
| Security: command proxy (sandboxed/paranoid) | openrouter | Working |
| Security: allowlist-only mode (paranoid) | openrouter | Working |
| Security: safe commands skip confirmation | openrouter | Working |
| Security: audit logging (JSONL) | openrouter | Working |
| `--no-security` / `--security-profile` flags | all | Working |

### Untested Features

| Feature | Reason |
|---------|--------|
| Claude API backend (`-b claude`) | Requires `pip install anthropic` |
| OpenAI API backend (`-b openai`) | Requires `pip install openai` |
| Ollama API backend (`-b ollama-api`) | Requires Ollama server running |
| Ollama CLI backend (`-b ollama-cli`) | Requires Ollama installed |
| Streaming (`--stream`) | Tested with OpenRouter |
| Cost tracking (`/cost`) | Requires API backend |
| Session save/load | Not tested in this session |
| Export (`/export`) | Not tested in this session |
| Search (`--search`) | Not tested in this session |
| Skills loading | Not tested in this session |
| Prompt templates | Not tested in this session |
| Command aliases | Not tested in this session |
| Multi-line input | Tested (paste detection + triggers) |
| Shell completions | Not tested in this session |

### Stream-JSON Support

Live tool call progress (shown in `--fancy` spinner) is supported by backends that output streaming JSON:

| Backend | Stream-JSON | Notes |
|---------|------------|-------|
| claude-code | Yes | `--output-format stream-json --verbose` |
| gemini | Yes | `-o stream-json` |
| coro | No | No structured JSON output available |
| claude (API) | N/A | API backends don't use CLI streaming |
| openrouter (API) | N/A | Direct API calls, full context control |
| openai (API) | N/A | API backends don't use CLI streaming |
| ollama-api | N/A | REST API, not CLI |
| ollama-cli | No | No structured JSON output available |

### Tool Handling Note

The CLI backends (coro, claude-code, gemini) manage their own tool execution internally. Our tool parsing (`tools.py`) is designed for:
- API backends that return structured tool calls (claude, openai, openrouter)
- Future CLI backends that output tool calls without executing them

For CLI backends, the agent loop provides value through:
- Context management across calls
- Session persistence
- Unified interface across backends
- Configuration and agent variants

## Troubleshooting

### "Command not found: coro"

Install the coro CLI or use a different backend:
```bash
python3 agent_loop.py -b claude "prompt"
```

### "ANTHROPIC_API_KEY not set"

Export your API key:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Tool execution stuck

Use Ctrl+C to interrupt, or set iteration limits:
```bash
python3 agent_loop.py -i 5 "prompt"  # Max 5 tool iterations
```

## License

MIT License - Part of UnifyWeaver
