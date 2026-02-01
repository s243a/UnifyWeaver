# Proposal: Terminal Diagnostics & Compatibility Suite

## Problem Statement

TUI applications (like coro-code) that use raw ANSI escape codes (`\r`, `\x1B[`) for cursor control don't work properly in Termux. The cursor doesn't return to line start, causing spinner animations and progress indicators to stack as separate lines instead of updating in place.

However, ncurses/terminfo-based cursor control (`tput cr`) works correctly. This indicates Termux requires terminfo queries rather than hardcoded escape sequences.

## Goals

1. **Diagnose** - Tools to observe and analyze terminal output issues
2. **Workaround** - Filter noisy output for immediate usability
3. **Fix** - Wrapper that translates raw escapes to terminfo calls
4. **Upstream** - File issues with coro-code and Termux
5. **Alternative** - Browser-based virtual terminal with proper escape handling

## Deliverables

### 1. mcp-terminal-diag (Done - Prototype)

MCP server for Claude Code to observe terminal output.

**Status:** Prototype complete at `tools/mcp-terminal-diag/server.py`

**Tools:**
- `tmux_capture_pane` - Capture terminal output
- `find_noise_patterns` - Analyze for spinner spam, token reports
- `tmux_search_pane` - Search patterns in output

### 2. coro-quiet Filter (Done - Basic)

Python wrapper that filters coro-code output.

**Status:** Basic version at `tools/coro-quiet/`

**Current features:**
- Collapse consecutive blank lines
- Capture token/cost stats, show summary at end
- Pass-through to coro CLI

**Needed:**
- [ ] Add `--terminfo` mode that translates escapes to tput calls
- [ ] Add spinner detection and collapse
- [ ] Test with coro interactive mode

### 3. terminfo-wrapper (New)

Wrapper/filter that translates raw ANSI escapes to terminfo calls.

**Approach:**
```python
# Intercept raw escapes and translate
TRANSLATIONS = {
    '\r': lambda: subprocess.run(['tput', 'cr']),
    '\x1B[K': lambda: subprocess.run(['tput', 'el']),
    '\x1B[1A': lambda: subprocess.run(['tput', 'cuu1']),
    '\x1B[2K': lambda: subprocess.run(['tput', 'el2']),
}
```

**Options:**
- A) Standalone wrapper: `terminfo-wrap coro`
- B) Integrate into coro-quiet: `coro-quiet --terminfo`
- C) PTY proxy that intercepts and translates in real-time

**Challenges:**
- Need to parse escape sequences from stream
- Must handle partial sequences at buffer boundaries
- Performance overhead of spawning tput for each escape

**Better approach - cache tput output:**
```python
import subprocess

# Pre-cache terminfo sequences at startup
TERMINFO_CACHE = {
    'cr': subprocess.run(['tput', 'cr'], capture_output=True).stdout,
    'el': subprocess.run(['tput', 'el'], capture_output=True).stdout,
    'cuu1': subprocess.run(['tput', 'cuu1'], capture_output=True).stdout,
}

def translate_escapes(data):
    # Replace raw escapes with cached terminfo sequences
    data = data.replace(b'\r', TERMINFO_CACHE['cr'])
    data = data.replace(b'\x1B[K', TERMINFO_CACHE['el'])
    return data
```

### 4. GitHub Issues

#### 4a. coro-code Issue

**Title:** Termux compatibility: use terminfo for cursor control instead of raw escapes

**Body:**
```markdown
## Environment
- Termux on Android
- TERM=xterm-256color
- coro 0.0.8

## Problem
The TUI spinner and progress indicators don't update in place. Instead of
overwriting the current line, each update creates a new line, causing
hundreds of lines of spinner spam.

## Root Cause
coro uses raw ANSI escape codes (`\r`, `\x1B[`) for cursor control.
Termux's terminal emulator requires terminfo-based sequences instead.

Test showing the difference:
```bash
# Raw \r - BROKEN in Termux (creates new lines)
for i in 1 2 3 4 5; do printf "\r Count: $i"; sleep 0.5; done

# tput cr - WORKS in Termux (updates in place)
for i in 1 2 3 4 5; do tput cr; printf " Count: $i"; sleep 0.5; done
```

## Suggested Fix
Use a terminal library that queries terminfo (like `crossterm` with proper
terminfo support, or `termion`) instead of hardcoding escape sequences.

Alternatively, provide a `--no-tui` or `--plain` mode for environments
where the TUI doesn't work properly.
```

#### 4b. Termux Issue

**Title:** Raw carriage return (\r) doesn't move cursor to line start

**Body:**
```markdown
## Environment
- Termux version: [version]
- Android version: [version]
- TERM=xterm-256color

## Problem
Raw carriage return (`\r`) and ANSI escape sequences like `\x1B[1G` don't
move the cursor to the beginning of the line. Instead, output continues
on a new line.

## Steps to Reproduce
```bash
for i in 1 2 3 4 5; do printf "\r Count: $i"; sleep 0.5; done; echo
```

Expected: Single line showing "Count: 5" (each iteration overwrites)
Actual: 5 separate lines

## Workaround
Using `tput cr` works correctly:
```bash
for i in 1 2 3 4 5; do tput cr; printf " Count: $i"; sleep 0.5; done; echo
```

## Impact
Many TUI applications (spinners, progress bars, interactive prompts)
that use raw escapes don't render correctly in Termux.

## Environment Details
```bash
echo $TERM        # xterm-256color
stty -a | grep cr # shows -ocrnl (correct)
infocmp | grep cr # terminfo entry
```
```

### 5. Browser Virtual Terminal (Future)

Web-based terminal emulator with proper escape code handling.

**Approach:**
- Use xterm.js in browser
- Connect to Termux via WebSocket
- Full escape sequence support

**Integration with UnifyWeaver:**
- Could be Prolog-generated like other web apps
- MCP server provides terminal data
- Fits the existing http-cli-server pattern

**Scope:** Larger project, defer to later phase

## Implementation Plan

### Phase 1: Immediate (This Session)
- [x] Create mcp-terminal-diag prototype
- [x] Create coro-quiet basic filter
- [ ] Commit current work to feature branch
- [ ] Write GitHub issue drafts

### Phase 2: terminfo-wrapper
- [ ] Design escape sequence parser
- [ ] Implement tput caching
- [ ] Create PTY proxy wrapper
- [ ] Add `--terminfo` flag to coro-quiet
- [ ] Test with coro interactive mode

### Phase 3: Upstream
- [ ] File coro-code GitHub issue
- [ ] File Termux GitHub issue
- [ ] Monitor responses, iterate on workarounds

### Phase 4: Browser Terminal (Future)
- [ ] Research xterm.js + WebSocket approach
- [ ] Design Prolog generator for terminal web app
- [ ] Implement and test

## Files

| Path | Description | Status |
|------|-------------|--------|
| `tools/coro-quiet/quiet_module.pl` | Prolog generator for filter | Done |
| `tools/coro-quiet/coro-quiet.py` | Generated Python wrapper | Done |
| `tools/mcp-terminal-diag/server.py` | MCP diagnostics server | Prototype |
| `tools/mcp-terminal-diag/PROPOSAL.md` | This document | Done |
| `tools/terminfo-wrapper/` | Escape→terminfo translator | TODO |

## Success Criteria

1. coro-code interactive mode usable in Termux (via wrapper or upstream fix)
2. Spinner/progress noise reduced to single updating line or filtered summary
3. MCP server allows Claude Code to diagnose terminal issues
4. Issues filed with coro-code and Termux
