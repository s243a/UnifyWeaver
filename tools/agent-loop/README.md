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
| `sessions.py` | Session save/load |
| `export.py` | Export to various formats |
| `costs.py` | API cost tracking |
| `search.py` | Session search |
| `retry.py` | Retry logic with backoff |
| `aliases.py` | Command aliases |
| `templates.py` | Prompt templates |
| `history.py` | History management |
| `multiline.py` | Multi-line input handling |
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
