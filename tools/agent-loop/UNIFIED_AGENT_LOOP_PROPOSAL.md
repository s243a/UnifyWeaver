# Proposal: UnifyWeaver Agent Loop

## Problem Statement

Interactive TUI-based coding agents (like coro-code) don't work properly in Termux and many terminal environments due to escape code / cursor control issues. Even xterm.js in browsers shows broken rendering.

However, these agents work fine in **single-task mode** where you pass a task as an argument and get text output. We can build our own agent loop that:
1. Calls agents in single-task mode repeatedly
2. Manages conversation context ourselves
3. Parses and handles tool calls
4. Provides clean, simple output

## Goals

1. **Bypass broken TUIs** - Use single-task mode instead of interactive
2. **Context management** - Maintain conversation history across calls
3. **Tool call handling** - Parse and execute tool calls from agent output
4. **Backend agnostic** - Same interface for coro, Claude API, OpenAI, etc.
5. **Prolog-generated** - Define agents declaratively, generate the loop

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
├───────────┬───────────┬───────────┬───────────┬─────────────┤
│   Coro    │  Claude   │  OpenAI   │  Ollama   │   Custom    │
│   (CLI)   │  (API)    │  (API)    │  (API)    │   (Plugin)  │
└───────────┴───────────┴───────────┴───────────┴─────────────┘
```

## Components

### 1. Agent Backend Interface

```python
class AgentBackend:
    """Abstract interface for agent backends."""

    def send_message(self, message: str, context: list[dict]) -> AgentResponse:
        """Send a message with context, get response."""
        raise NotImplementedError

    def parse_tool_calls(self, response: str) -> list[ToolCall]:
        """Extract tool calls from response."""
        raise NotImplementedError

    def supports_streaming(self) -> bool:
        """Whether this backend supports streaming output."""
        return False
```

### 2. Coro Backend (CLI)

```python
class CoroBackend(AgentBackend):
    """Coro-code CLI backend using single-task mode."""

    def send_message(self, message: str, context: list[dict]) -> AgentResponse:
        # Format context into prompt
        prompt = self.format_prompt(message, context)

        # Call coro in single-task mode
        result = subprocess.run(
            ['coro', '--verbose', prompt],
            capture_output=True, text=True
        )

        return AgentResponse(
            content=self.clean_output(result.stdout),
            tool_calls=self.parse_tool_calls(result.stdout),
            tokens=self.parse_tokens(result.stdout)
        )

    def format_prompt(self, message: str, context: list[dict]) -> str:
        """Format message with conversation context."""
        if not context:
            return message

        history = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in context[-5:]  # Last 5 messages
        ])

        return f"""Previous conversation:
{history}

Current request: {message}"""
```

### 3. Claude API Backend

```python
class ClaudeBackend(AgentBackend):
    """Anthropic Claude API backend."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def send_message(self, message: str, context: list[dict]) -> AgentResponse:
        messages = context + [{"role": "user", "content": message}]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=messages
        )

        return AgentResponse(
            content=response.content[0].text,
            tool_calls=self.extract_tool_use(response),
            tokens={'input': response.usage.input_tokens,
                    'output': response.usage.output_tokens}
        )
```

### 4. Context Manager

```python
class ContextManager:
    """Manages conversation history and context window."""

    def __init__(self, max_tokens: int = 100000):
        self.messages: list[dict] = []
        self.max_tokens = max_tokens
        self.token_count = 0

    def add_message(self, role: str, content: str, tokens: int = 0):
        """Add message to context."""
        self.messages.append({
            'role': role,
            'content': content,
            'tokens': tokens
        })
        self.token_count += tokens
        self._trim_if_needed()

    def _trim_if_needed(self):
        """Remove old messages if context too large."""
        while self.token_count > self.max_tokens and len(self.messages) > 2:
            removed = self.messages.pop(0)
            self.token_count -= removed.get('tokens', 0)

    def get_context(self) -> list[dict]:
        """Get context for next request."""
        return [{'role': m['role'], 'content': m['content']}
                for m in self.messages]
```

### 5. Tool Call Handler

```python
class ToolHandler:
    """Handles tool calls from agent responses."""

    def __init__(self, allowed_tools: list[str] = None):
        self.allowed_tools = allowed_tools or ['bash', 'read', 'write', 'edit']
        self.tools = {
            'bash': self.execute_bash,
            'read': self.read_file,
            'write': self.write_file,
            'edit': self.edit_file,
        }

    def execute(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return result."""
        if tool_call.name not in self.allowed_tools:
            return f"Tool '{tool_call.name}' not allowed"

        if tool_call.name not in self.tools:
            return f"Unknown tool: {tool_call.name}"

        return self.tools[tool_call.name](tool_call.arguments)

    def execute_bash(self, args: dict) -> str:
        """Execute a bash command."""
        cmd = args.get('command', '')
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout + result.stderr
```

### 6. Agent Loop

```python
class AgentLoop:
    """Main agent loop orchestrating all components."""

    def __init__(self, backend: AgentBackend, tools: ToolHandler = None):
        self.backend = backend
        self.context = ContextManager()
        self.tools = tools or ToolHandler()
        self.running = True

    def run(self):
        """Main loop - read input, get response, handle tools."""
        print("UnifyWeaver Agent Loop")
        print(f"Backend: {self.backend.__class__.__name__}")
        print("Type 'exit' to quit, 'clear' to reset context\n")

        while self.running:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue
                if user_input.lower() == 'exit':
                    break
                if user_input.lower() == 'clear':
                    self.context = ContextManager()
                    print("Context cleared.\n")
                    continue

                # Add user message to context
                self.context.add_message('user', user_input)

                # Get agent response
                response = self.backend.send_message(
                    user_input,
                    self.context.get_context()
                )

                # Handle tool calls
                while response.tool_calls:
                    tool_results = []
                    for tool_call in response.tool_calls:
                        print(f"  [Tool: {tool_call.name}]")
                        result = self.tools.execute(tool_call)
                        tool_results.append(result)

                    # Continue with tool results
                    response = self.backend.send_message(
                        f"Tool results:\n{tool_results}",
                        self.context.get_context()
                    )

                # Display response
                print(f"\nAssistant: {response.content}\n")

                # Add to context
                self.context.add_message('assistant', response.content,
                                         response.tokens.get('output', 0))

                # Show token summary
                if response.tokens:
                    print(f"  [Tokens: {response.tokens}]\n")

            except KeyboardInterrupt:
                print("\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"Error: {e}\n")
```

## Prolog Specification (UnifyWeaver Style)

```prolog
:- module(agent_loop_spec, [agent_backend/2, tool_spec/2]).

%% Agent Backend Definitions

agent_backend(coro, [
    type(cli),
    command("coro"),
    args(["--verbose"]),
    context_format(conversation_history),
    output_parser(coro_parser),
    tool_pattern("Tool call: (\\w+)\\((.*)\\)"),
    token_pattern("Tokens: (\\d+) input \\+ (\\d+) output")
]).

agent_backend(claude_api, [
    type(api),
    endpoint("https://api.anthropic.com/v1/messages"),
    model("claude-sonnet-4-20250514"),
    auth_header("x-api-key"),
    context_format(messages_array),
    supports_tools(true),
    supports_streaming(true)
]).

agent_backend(openai, [
    type(api),
    endpoint("https://api.openai.com/v1/chat/completions"),
    model("gpt-4o"),
    auth_header("Authorization", "Bearer"),
    context_format(messages_array),
    supports_tools(true)
]).

agent_backend(ollama, [
    type(api),
    endpoint("http://localhost:11434/api/chat"),
    model("llama3"),
    context_format(messages_array),
    auth_required(false)
]).

%% Tool Definitions

tool_spec(bash, [
    description("Execute a bash command"),
    parameters([
        param(command, string, required, "The command to execute")
    ]),
    confirmation_required(true)
]).

tool_spec(read, [
    description("Read a file"),
    parameters([
        param(path, string, required, "Path to file")
    ])
]).

tool_spec(write, [
    description("Write content to a file"),
    parameters([
        param(path, string, required, "Path to file"),
        param(content, string, required, "Content to write")
    ]),
    confirmation_required(true)
]).

tool_spec(edit, [
    description("Edit a file with search/replace"),
    parameters([
        param(path, string, required, "Path to file"),
        param(old_string, string, required, "Text to find"),
        param(new_string, string, required, "Replacement text")
    ]),
    confirmation_required(true)
]).

%% Loop Configuration

loop_config([
    max_context_tokens(100000),
    trim_strategy(oldest_first),
    display_tokens(true),
    display_tool_calls(true),
    confirm_destructive_tools(true)
]).
```

## File Structure

```
tools/agent-loop/
├── PROPOSAL.md              # This document
├── agent_loop_module.pl     # Prolog generator
├── generated/
│   ├── agent_loop.py        # Main loop (generated)
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract backend
│   │   ├── coro.py          # Coro CLI backend
│   │   ├── claude.py        # Claude API backend
│   │   └── openai.py        # OpenAI API backend
│   ├── context.py           # Context manager
│   ├── tools.py             # Tool handler
│   └── display.py           # Output renderer
└── README.md                # Generated documentation
```

## Termux Compatibility Notes

### What Works
- `tput cr` (terminfo carriage return) - cursor moves to line start
- `tput el` (terminfo clear to end of line)
- ncurses-based apps like `dialog` - full TUI works
- Basic ANSI colors and formatting

### What Doesn't Work
- Raw `\r` (carriage return) - creates new line instead of returning
- Raw `\x1B[1G` (cursor to column 1) - doesn't work
- Raw `\x1B[K` (clear line) - doesn't work
- Coro-code's TUI - uses raw escapes, completely broken
- xterm.js in browser - also shows broken coro TUI (sizing issues)

### Root Cause
Termux requires terminfo-based terminal control (`tput`) rather than
hardcoded ANSI escape sequences. Apps using raw escapes will break.

### Implication for Agent Loop
Start with **append-only mode** that uses no cursor control at all.
This guarantees compatibility. Add optional terminfo-based display
modes later for environments that support it.

---

## Display Modes

The agent loop supports multiple display modes, starting simple and adding complexity:

### Mode 1: Append-Only (Initial Implementation)

Simple terminal output with no cursor control. All output appends to the terminal.
Works everywhere - Termux, pipes, logs, dumb terminals, etc.

**Features:**
- Pure text output, no escape codes
- Each message appends below previous
- Progress shown as text lines
- Full conversation visible by scrolling

**Example session:**
```
UnifyWeaver Agent Loop
Backend: CoroBackend
Type 'exit' to quit, 'clear' to reset context

You: What is 2+2?

[Calling backend...]
Assistant: The answer is 4.

  [Tokens: {input: 42, output: 15}]

You: exit

Goodbye!
```

### Mode 2: ncurses-based (Future)

Uses ncurses/terminfo for cursor control. Works in Termux and proper terminals.

**Features:**
- Spinner that updates in place
- Progress bar for streaming
- Status line at bottom
- Uses \`tput\` for all cursor control

### Mode 3: Limited Context Display (Future)

Scrollable view showing only recent messages, with cursor position control.

**Features:**
- Show only last N lines of conversation
- Scroll through history with keys
- Input line stays at fixed position
- Requires ncurses Mode 2 as foundation

---

## Context Modes

Different ways to handle conversation context when sending to the backend:

### Context Behavior

| Mode | Description |
|------|-------------|
| `continue` | New query continues the conversation, uses full context |
| `fresh` | Each query is independent, no prior context |
| `summarize` | Summarize old context when it grows large |
| `sliding` | Keep only the last N messages |

### Context Format

How context is formatted when sent to the backend:

| Format | Use Case |
|--------|----------|
| `plain` | Simple text, each message on a line with role prefix |
| `markdown` | Markdown formatted with headers for each message |
| `json` | JSON array of message objects (for API backends) |
| `xml` | XML structure with message elements |

**Example plain format:**
```
User: What is 2+2?
Assistant: 4
User: Add 3 to that
```

---

## Implementation Plan

### Phase 1: Append-Only Prototype
- [ ] Basic agent loop with stdin/stdout
- [ ] Coro CLI backend (single-task mode)
- [ ] Context manager with message history
- [ ] Pure text output, no escape codes
- [ ] Test in Termux

### Phase 2: Claude API Backend
- [ ] Add Claude API backend
- [ ] API key configuration
- [ ] Token counting and display
- [ ] Streaming support (optional)

### Phase 3: Tool Handling
- [ ] Parse tool calls from coro output
- [ ] Execute bash, read, write, edit tools
- [ ] Confirmation prompts for destructive operations
- [ ] Tool result formatting and continuation

### Phase 4: Prolog Generation
- [ ] Create agent_loop_module.pl
- [ ] Define backend specs in Prolog
- [ ] Generate Python code from specs
- [ ] Generate documentation

### Phase 5: ncurses Display Mode (Future)
- [ ] Add ncurses/terminfo based display
- [ ] Spinner that updates in place
- [ ] Progress bar for streaming
- [ ] Status line at bottom

### Phase 6: Advanced Features (Future)
- [ ] OpenAI and Ollama backends
- [ ] Limited context display mode
- [ ] Context summarization
- [ ] Multiple context formats (plain, markdown, json, xml)
