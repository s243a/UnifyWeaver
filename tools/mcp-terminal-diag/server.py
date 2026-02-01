#!/usr/bin/env python3
"""MCP Terminal Diagnostics Server

Provides tools for Claude Code to observe terminal output for diagnostic purposes.
Supports multiple capture methods: tmux, script logs, and direct file tailing.

Usage:
  python server.py

Then configure in Claude Code's MCP settings.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# MCP protocol implementation (minimal STDIO transport)
class MCPServer:
    def __init__(self):
        self.tools = {}
        self.name = "terminal-diag"
        self.version = "0.1.0"

    def tool(self, name: str, description: str):
        """Decorator to register a tool"""
        def decorator(func):
            self.tools[name] = {
                "function": func,
                "description": description,
                "schema": getattr(func, "_schema", {})
            }
            return func
        return decorator

    async def handle_request(self, request: dict) -> dict:
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        if method == "initialize":
            return self._result(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": self.name, "version": self.version}
            })

        elif method == "tools/list":
            tools_list = [
                {
                    "name": name,
                    "description": info["description"],
                    "inputSchema": info["schema"]
                }
                for name, info in self.tools.items()
            ]
            return self._result(req_id, {"tools": tools_list})

        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})

            if tool_name not in self.tools:
                return self._error(req_id, -32601, f"Unknown tool: {tool_name}")

            try:
                result = await self.tools[tool_name]["function"](**args)
                return self._result(req_id, {
                    "content": [{"type": "text", "text": str(result)}]
                })
            except Exception as e:
                return self._error(req_id, -32000, str(e))

        elif method == "notifications/initialized":
            return None  # No response for notifications

        return self._error(req_id, -32601, f"Unknown method: {method}")

    def _result(self, req_id, result):
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _error(self, req_id, code, message):
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    async def run(self):
        """Run the MCP server on STDIO"""
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break

                request = json.loads(line)
                response = await self.handle_request(request)

                if response:
                    print(json.dumps(response), flush=True)

            except json.JSONDecodeError as e:
                print(json.dumps({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"}
                }), flush=True)
            except Exception as e:
                print(json.dumps({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32000, "message": str(e)}
                }), flush=True)


# Create server instance
server = MCPServer()


# =============================================================================
# TMUX-BASED DIAGNOSTICS
# =============================================================================

@server.tool("tmux_list_sessions", "List all tmux sessions and their panes")
async def tmux_list_sessions() -> str:
    """List all tmux sessions with their windows and panes."""
    try:
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}: #{session_windows} windows"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return f"No tmux sessions found or tmux not running: {result.stderr}"

        sessions = result.stdout.strip()

        # Get panes for each session
        panes_result = subprocess.run(
            ["tmux", "list-panes", "-a", "-F", "#{session_name}:#{window_index}.#{pane_index} - #{pane_current_command} (#{pane_width}x#{pane_height})"],
            capture_output=True, text=True, timeout=5
        )

        return f"Sessions:\n{sessions}\n\nPanes:\n{panes_result.stdout}"
    except FileNotFoundError:
        return "tmux not installed"
    except subprocess.TimeoutExpired:
        return "tmux command timed out"


@server.tool("tmux_capture_pane", "Capture recent output from a tmux pane")
async def tmux_capture_pane(
    target: str = "0",
    lines: int = 50,
    strip_ansi: bool = False
) -> str:
    """
    Capture the last N lines from a tmux pane.

    Args:
        target: Pane target (e.g., "0", "session:window.pane")
        lines: Number of lines to capture (default 50)
        strip_ansi: Remove ANSI escape codes for cleaner output
    """
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", target, "-p", "-S", f"-{lines}"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return f"Error capturing pane: {result.stderr}"

        output = result.stdout

        if strip_ansi:
            # Remove ANSI escape sequences
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            output = ansi_escape.sub('', output)

        return output
    except FileNotFoundError:
        return "tmux not installed"
    except subprocess.TimeoutExpired:
        return "tmux command timed out"


@server.tool("tmux_search_pane", "Search for a pattern in tmux pane output")
async def tmux_search_pane(
    pattern: str,
    target: str = "0",
    lines: int = 500,
    context: int = 2
) -> str:
    """
    Search for a regex pattern in tmux pane output.

    Args:
        pattern: Regex pattern to search for
        target: Pane target
        lines: How many lines of history to search
        context: Lines of context around matches
    """
    try:
        # Capture pane content
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", target, "-p", "-S", f"-{lines}"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return f"Error: {result.stderr}"

        content_lines = result.stdout.split('\n')
        regex = re.compile(pattern, re.IGNORECASE)

        matches = []
        for i, line in enumerate(content_lines):
            if regex.search(line):
                start = max(0, i - context)
                end = min(len(content_lines), i + context + 1)
                snippet = '\n'.join(f"  {j}: {content_lines[j]}" for j in range(start, end))
                matches.append(f"Match at line {i}:\n{snippet}")

        if not matches:
            return f"No matches for pattern: {pattern}"

        return f"Found {len(matches)} matches:\n\n" + "\n\n".join(matches[:10])
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# SCRIPT LOG DIAGNOSTICS
# =============================================================================

SCRIPT_LOG_DIR = Path.home() / ".terminal-diag-logs"

@server.tool("script_start_logging", "Start logging a terminal session to a file")
async def script_start_logging(name: str = "session") -> str:
    """
    Provides instructions to start logging with the 'script' command.

    Args:
        name: Name for the log file
    """
    SCRIPT_LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = SCRIPT_LOG_DIR / f"{name}_{timestamp}.log"

    return f"""To start logging your terminal session, run:

  script -f {log_path}

This will record all terminal output. When done, type 'exit' to stop logging.

Then use 'script_read_log' tool to analyze the captured output."""


@server.tool("script_list_logs", "List available script log files")
async def script_list_logs() -> str:
    """List all terminal log files."""
    if not SCRIPT_LOG_DIR.exists():
        return "No log directory found. Use script_start_logging first."

    logs = sorted(SCRIPT_LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not logs:
        return "No log files found."

    result = ["Available logs:"]
    for log in logs[:10]:
        size = log.stat().st_size
        mtime = datetime.fromtimestamp(log.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        result.append(f"  {log.name} ({size:,} bytes, {mtime})")

    return '\n'.join(result)


@server.tool("script_read_log", "Read recent content from a script log file")
async def script_read_log(
    name: str,
    lines: int = 100,
    strip_ansi: bool = True
) -> str:
    """
    Read the last N lines from a script log file.

    Args:
        name: Log filename (from script_list_logs)
        lines: Number of lines to read
        strip_ansi: Remove ANSI codes for cleaner output
    """
    log_path = SCRIPT_LOG_DIR / name
    if not log_path.exists():
        return f"Log file not found: {name}"

    try:
        with open(log_path, 'r', errors='replace') as f:
            content = f.read()

        if strip_ansi:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            content = ansi_escape.sub('', content)

        content_lines = content.split('\n')
        return '\n'.join(content_lines[-lines:])
    except Exception as e:
        return f"Error reading log: {e}"


# =============================================================================
# DIRECT FILE TAILING
# =============================================================================

@server.tool("tail_file", "Read the last N lines of any file")
async def tail_file(
    path: str,
    lines: int = 50,
    strip_ansi: bool = False
) -> str:
    """
    Read the last N lines of a file (useful for log files).

    Args:
        path: Path to the file
        lines: Number of lines to read
        strip_ansi: Remove ANSI escape codes
    """
    try:
        file_path = Path(path).expanduser()
        if not file_path.exists():
            return f"File not found: {path}"

        with open(file_path, 'r', errors='replace') as f:
            content = f.read()

        if strip_ansi:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            content = ansi_escape.sub('', content)

        content_lines = content.split('\n')
        return '\n'.join(content_lines[-lines:])
    except Exception as e:
        return f"Error: {e}"


@server.tool("find_noise_patterns", "Analyze terminal output for common noise patterns")
async def find_noise_patterns(
    target: str = "0",
    lines: int = 200
) -> str:
    """
    Analyze tmux pane output and identify noise patterns like:
    - Repeated blank lines
    - Token usage reports
    - Progress indicators
    - Redundant status messages

    Args:
        target: tmux pane target
        lines: Lines of history to analyze
    """
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", target, "-p", "-S", f"-{lines}"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return f"Error: {result.stderr}"

        content = result.stdout
        content_lines = content.split('\n')

        analysis = {
            "total_lines": len(content_lines),
            "blank_lines": sum(1 for l in content_lines if not l.strip()),
            "consecutive_blanks": 0,
            "token_lines": [],
            "repeated_lines": {},
            "ansi_heavy_lines": 0
        }

        # Count consecutive blanks
        max_consecutive = 0
        current_consecutive = 0
        for line in content_lines:
            if not line.strip():
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        analysis["consecutive_blanks"] = max_consecutive

        # Find token/cost patterns
        token_patterns = [
            r'[Tt]okens?:?\s*\d+',
            r'[Cc]ost:?\s*\$',
            r'[Dd]uration:?\s*\d+',
            r'input\s*\+\s*\d+\s*output'
        ]
        for i, line in enumerate(content_lines):
            for pattern in token_patterns:
                if re.search(pattern, line):
                    analysis["token_lines"].append(f"L{i}: {line[:80]}")
                    break

        # Find repeated lines
        for line in content_lines:
            stripped = line.strip()
            if stripped and len(stripped) > 10:
                analysis["repeated_lines"][stripped] = analysis["repeated_lines"].get(stripped, 0) + 1

        # Filter to only repeated ones
        analysis["repeated_lines"] = {k: v for k, v in analysis["repeated_lines"].items() if v > 1}

        # Count ANSI-heavy lines
        ansi_pattern = re.compile(r'\x1B\[')
        for line in content_lines:
            if len(ansi_pattern.findall(line)) > 5:
                analysis["ansi_heavy_lines"] += 1

        # Format report
        report = [
            "=== Terminal Noise Analysis ===",
            f"Total lines: {analysis['total_lines']}",
            f"Blank lines: {analysis['blank_lines']} ({100*analysis['blank_lines']//max(1,analysis['total_lines'])}%)",
            f"Max consecutive blanks: {analysis['consecutive_blanks']}",
            f"ANSI-heavy lines: {analysis['ansi_heavy_lines']}",
            "",
            f"Token/cost mentions ({len(analysis['token_lines'])}):"
        ]
        for tl in analysis["token_lines"][:5]:
            report.append(f"  {tl}")

        if analysis["repeated_lines"]:
            report.append(f"\nRepeated lines ({len(analysis['repeated_lines'])}):")
            for line, count in sorted(analysis["repeated_lines"].items(), key=lambda x: -x[1])[:5]:
                report.append(f"  {count}x: {line[:60]}...")

        return '\n'.join(report)
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    asyncio.run(server.run())
