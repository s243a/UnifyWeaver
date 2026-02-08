# Add automatic paste detection for multi-line input

## Problem

When a user pastes multi-line text into the agent loop's interactive prompt, only the first line is captured. The remaining lines are either lost or fed into subsequent prompts as separate inputs.

Current workarounds require the user to know a trigger syntax before pasting:

| Syntax | Mode |
|--------|------|
| ` ``` ` | Code block (read until closing ` ``` `) |
| `<<<` or `<<<MARKER` | Heredoc (read until marker) |
| Trailing `\` | Line continuation |
| `{`, `[`, `(` | Data structure (read until matching close) |

These work but are not discoverable. A user pasting a stack trace, a code snippet, or a multi-paragraph question will lose content without warning.

## Solution: Timing-based paste detection

After `input()` returns the first line, immediately check if more data is available on stdin using `select.select()` with a short timeout (50ms). If data arrives within that window, it's a paste — keep reading lines until no more data arrives within the timeout.

This works because:
- **Typing**: A human types one line, pauses, presses Enter. No data arrives within 50ms after the line.
- **Pasting**: The terminal sends all lines in rapid succession. Each subsequent line is available on stdin within microseconds of the previous one.

### How it works

```python
import select
import sys

def _read_pasted_lines(first_line: str, timeout: float = 0.05) -> str:
    """After reading first_line via input(), check for pasted continuation lines."""
    lines = [first_line]
    while True:
        # Check if more data is available on stdin
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            break  # No more data within timeout — paste is complete
        line = sys.stdin.readline()
        if not line:
            break  # EOF
        lines.append(line.rstrip('\n'))
    return '\n'.join(lines)
```

### Integration point

The change is entirely within `multiline.py`'s `get_input_smart()` function. After `input(prompt)` returns a line that doesn't match any existing trigger, call `_read_pasted_lines()` to check for continuation:

```python
def get_input_smart(prompt: str = "You: ") -> str | None:
    try:
        line = input(prompt)
    except EOFError:
        return None

    if not line:
        return ""

    stripped = line.strip()

    # Existing triggers: ```, <<<, \, {/[/( ...
    # (unchanged)

    # NEW: check for pasted continuation lines
    if sys.stdin.isatty():
        return _read_pasted_lines(line)

    return line
```

The `sys.stdin.isatty()` check ensures we only do paste detection on a real terminal. When stdin is piped (e.g., `echo "prompt" | python3 agent_loop.py`), we skip detection and behave as before.

### Tradeoffs

| Aspect | Impact |
|--------|--------|
| Latency | 50ms per input (imperceptible to humans) |
| Accuracy | Excellent on local terminals; may miss very slow paste on laggy SSH |
| Compatibility | `select.select()` works on all Unix (Linux, macOS, Termux). Not available on Windows. |
| Fallback | On Windows or non-tty stdin, falls back to single-line (existing behavior) |
| Existing triggers | Unchanged — ` ``` `, `<<<`, `\` still work as before |

### Edge cases

- **Slow SSH connection**: Paste might arrive with >50ms gaps between lines. Could increase timeout, but that adds latency to every input. 50ms is a good default; can be made configurable.
- **Single-line paste**: Works fine — no continuation data, returns the single line.
- **Empty lines in paste**: Captured correctly — `readline()` returns `'\n'` for empty lines, not `''`.
- **Paste ending with newline**: Terminal paste typically ends with a newline, which triggers `input()` to return. The `select` check then finds no more data.

## Files to modify

| File | Change |
|------|--------|
| `tools/agent-loop/generated/multiline.py` | Add `_read_pasted_lines()`, integrate into `get_input_smart()` |

No changes needed to `agent_loop.py` — it already calls `get_input_smart()`.

## Future: Bracketed paste mode

A more robust approach is terminal bracketed paste, where the terminal wraps pasted content in escape sequences (`\e[200~` ... `\e[201~`). This eliminates timing ambiguity entirely.

However, it requires:
- Raw terminal mode (`tty.setraw`), which disables readline line editing
- Reimplementing backspace, arrow keys, history, etc. — or using a library like `prompt_toolkit`
- Careful cleanup on exit/crash to restore terminal state
- Fallback for terminals that don't support it

This could be added as an opt-in mode (`--bracketed-paste`) in a future iteration, potentially using `prompt_toolkit` if it's available.

## Test plan

- [ ] Paste a multi-line code snippet — all lines captured
- [ ] Paste a stack trace — all lines captured
- [ ] Type a single line normally — no delay, works as before
- [ ] Pipe input (`echo "test" | python3 agent_loop.py`) — no regression
- [ ] Existing triggers still work: ` ``` `, `<<<`, `\`
- [ ] Interactive `/multiline` toggle still works

## Verification

```bash
cd tools/agent-loop/generated

# Interactive mode — paste a multi-line block
python3 agent_loop.py -b openrouter
# Then paste:
#   What does this code do?
#   def fib(n):
#       if n <= 1: return n
#       return fib(n-1) + fib(n-2)

# Verify all 4 lines are captured as a single prompt

# Pipe mode (should not hang waiting for paste)
echo "What is 2+2?" | python3 agent_loop.py -b openrouter
```

---

Generated with [Claude Code](https://claude.com/claude-code)
