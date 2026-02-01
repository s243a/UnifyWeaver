# MCP Terminal Diagnostics

MCP server that lets Claude Code observe terminal output for diagnostic purposes.

## Tools

### tmux-based (real-time)
- `tmux_list_sessions` - List all tmux sessions and panes
- `tmux_capture_pane` - Capture last N lines from a pane
- `tmux_search_pane` - Search for patterns in pane output
- `find_noise_patterns` - Analyze output for common noise (blank lines, token reports, etc.)

### script-based (persistent logs)
- `script_start_logging` - Instructions to start logging with `script`
- `script_list_logs` - List available log files
- `script_read_log` - Read from a log file

### file-based
- `tail_file` - Read last N lines of any file

## Setup

### 1. Start a tmux session for your work

```bash
tmux new -s work
# Run coro or other commands here
```

### 2. Configure Claude Code MCP

Add to your Claude Code MCP config:

```json
{
  "mcpServers": {
    "terminal-diag": {
      "command": "python",
      "args": ["/path/to/tools/mcp-terminal-diag/server.py"]
    }
  }
}
```

### 3. Use from Claude Code

```
# List available panes
> Use terminal-diag to list tmux sessions

# Capture recent output
> Capture the last 100 lines from tmux pane 0

# Analyze for noise
> Find noise patterns in my terminal
```

## Example Workflow

1. Start tmux: `tmux new -s coro`
2. Run coro interactive: `coro`
3. Have a conversation, notice noise
4. Ask Claude Code: "Analyze the noise in my coro session"
5. Claude calls `find_noise_patterns` and sees exactly what patterns appear

## Privacy Note

This server can read terminal output. Only enable it when you need diagnostics, and be mindful of sensitive data in your terminal history.
