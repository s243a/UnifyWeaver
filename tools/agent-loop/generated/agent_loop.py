#!/usr/bin/env python3
"""
UnifyWeaver Agent Loop - Append-Only Mode

A simple agent loop that works in any terminal environment.
Uses pure text output with no cursor control or escape codes.
"""

import sys
import argparse
from pathlib import Path
from backends import AgentBackend, CoroBackend
from context import ContextManager, ContextBehavior
from tools import ToolHandler
from config import load_config, load_config_from_dir, get_default_config, AgentConfig, Config, save_example_config


class AgentLoop:
    """Main agent loop orchestrating all components."""

    def __init__(
        self,
        backend: AgentBackend,
        context: ContextManager | None = None,
        tools: ToolHandler | None = None,
        show_tokens: bool = True,
        auto_execute_tools: bool = False,
        max_iterations: int = 0,
        config: Config | None = None
    ):
        self.backend = backend
        self.context = context or ContextManager()
        self.tools = tools or ToolHandler()
        self.show_tokens = show_tokens
        self.auto_execute_tools = auto_execute_tools
        self.max_iterations = max_iterations  # 0 = unlimited
        self.config = config  # For backend switching
        self.running = True

    def run(self) -> None:
        """Main loop - read input, get response, display output."""
        self._print_header()

        while self.running:
            try:
                # Get user input
                user_input = self._get_input()

                if user_input is None:
                    # EOF
                    break

                if not user_input.strip():
                    continue

                # Handle commands
                if self._handle_command(user_input):
                    continue

                # Process the message
                self._process_message(user_input)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"\n[Error: {e}]\n")

        print("\nGoodbye!")

    def _print_header(self) -> None:
        """Print the startup header."""
        print("UnifyWeaver Agent Loop")
        print(f"Backend: {self.backend.name}")
        iterations_str = "unlimited" if self.max_iterations == 0 else str(self.max_iterations)
        print(f"Max iterations: {iterations_str}")
        print("Type 'exit' to quit, '/help' for commands")
        print()

    def _get_input(self) -> str | None:
        """Get input from user."""
        try:
            return input("You: ")
        except EOFError:
            return None

    def _handle_command(self, user_input: str) -> bool:
        """Handle special commands. Returns True if command was handled."""
        text = user_input.strip()
        cmd = text.lower()

        # Handle both with and without slash prefix
        if cmd.startswith('/'):
            cmd = cmd[1:]
            text = text[1:]

        if cmd == 'exit' or cmd == 'quit':
            self.running = False
            return True

        if cmd == 'clear':
            self.context.clear()
            print("[Context cleared]\n")
            return True

        if cmd == 'help':
            self._print_help()
            return True

        if cmd == 'status':
            self._print_status()
            return True

        # /iterations N - set max iterations
        if cmd.startswith('iterations'):
            return self._handle_iterations_command(text)

        # /backend <name> - switch backend
        if cmd.startswith('backend'):
            return self._handle_backend_command(text)

        return False

    def _handle_iterations_command(self, text: str) -> bool:
        """Handle /iterations N command."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            iterations_str = "unlimited" if self.max_iterations == 0 else str(self.max_iterations)
            print(f"[Current max iterations: {iterations_str}]")
            print("[Usage: /iterations N (0 = unlimited)]\n")
            return True

        try:
            n = int(parts[1])
            if n < 0:
                raise ValueError("Must be >= 0")
            self.max_iterations = n
            iterations_str = "unlimited" if n == 0 else str(n)
            print(f"[Max iterations set to: {iterations_str}]\n")
        except ValueError as e:
            print(f"[Error: Invalid number - {e}]\n")
        return True

    def _handle_backend_command(self, text: str) -> bool:
        """Handle /backend <name> command."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            print(f"[Current backend: {self.backend.name}]")
            if self.config:
                print(f"[Available: {', '.join(self.config.agents.keys())}]")
            print("[Usage: /backend <agent-name>]\n")
            return True

        agent_name = parts[1].strip()

        if not self.config:
            print("[Error: No config loaded, cannot switch backends]\n")
            return True

        if agent_name not in self.config.agents:
            print(f"[Error: Unknown agent '{agent_name}']")
            print(f"[Available: {', '.join(self.config.agents.keys())}]\n")
            return True

        try:
            agent_config = self.config.agents[agent_name]
            self.backend = create_backend_from_config(agent_config)
            print(f"[Switched to backend: {self.backend.name}]\n")
        except Exception as e:
            print(f"[Error switching backend: {e}]\n")
        return True

    def _print_help(self) -> None:
        """Print help message."""
        print("""
Commands (with or without / prefix):
  /exit, /quit       - Exit the agent loop
  /clear             - Clear conversation context
  /status            - Show context status
  /help              - Show this help message
  /iterations [N]    - Show or set max iterations (0 = unlimited)
  /backend [name]    - Show or switch backend/agent

Just type your message to chat with the agent.
""")

    def _print_status(self) -> None:
        """Print context status."""
        iterations_str = "unlimited" if self.max_iterations == 0 else str(self.max_iterations)
        print(f"""
Status:
  Backend: {self.backend.name}
  Max iterations: {iterations_str}
  Messages: {self.context.message_count}
  Tokens: {self.context.token_count}
""")

    def _process_message(self, user_input: str) -> None:
        """Process a user message and get response."""
        # Add to context
        self.context.add_message('user', user_input)

        # Show that we're calling the backend
        print("\n[Calling backend...]")

        # Get response from backend
        response = self.backend.send_message(
            user_input,
            self.context.get_context()
        )

        # Handle tool calls if present
        iteration_count = 0
        while response.tool_calls:
            iteration_count += 1

            # Check iteration limit
            if self.max_iterations > 0 and iteration_count > self.max_iterations:
                print(f"\n[Paused after {self.max_iterations} iteration(s)]")
                print("[Press Enter to continue, or type a message]")
                try:
                    cont = input("You: ")
                    if cont.strip():
                        # User provided new input - process it instead
                        self.context.add_message('assistant', response.content, 0)
                        self._process_message(cont)
                        return
                    # Reset counter and continue
                    iteration_count = 1
                except EOFError:
                    return

            tool_results = []

            for tool_call in response.tool_calls:
                # Execute the tool (with confirmation unless auto_execute)
                if self.auto_execute_tools:
                    # Skip confirmation
                    self.tools.confirm_destructive = False

                result = self.tools.execute(tool_call)
                tool_results.append(self.tools.format_result_for_agent(result))

                # Show result to user
                print(f"  {result.output[:200]}{'...' if len(result.output) > 200 else ''}")

            # Send tool results back to agent for continuation
            tool_message = "\n".join(tool_results)
            self.context.add_message('user', f"Tool results:\n{tool_message}")

            print(f"\n[Iteration {iteration_count}: continuing with tool results...]")
            response = self.backend.send_message(
                f"Tool results:\n{tool_message}",
                self.context.get_context()
            )

        # Display response
        print(f"\nAssistant: {response.content}\n")

        # Add response to context
        output_tokens = response.tokens.get('output', 0)
        self.context.add_message('assistant', response.content, output_tokens)

        # Show token summary if enabled
        if self.show_tokens and response.tokens:
            token_info = ", ".join(
                f"{k}: {v}" for k, v in response.tokens.items()
            )
            print(f"  [Tokens: {token_info}]\n")


def create_backend_from_config(agent_config: AgentConfig) -> AgentBackend:
    """Create a backend from an AgentConfig."""
    backend_type = agent_config.backend

    if backend_type == 'coro':
        return CoroBackend(command=agent_config.command or 'claude')

    elif backend_type == 'claude-code':
        from backends import ClaudeCodeBackend
        return ClaudeCodeBackend(
            command=agent_config.command or 'claude',
            model=agent_config.model or 'sonnet'
        )

    elif backend_type == 'gemini':
        from backends import GeminiBackend
        return GeminiBackend(
            command=agent_config.command or 'gemini',
            model=agent_config.model or 'gemini-2.5-flash'
        )

    elif backend_type == 'claude':
        from backends import ClaudeAPIBackend
        return ClaudeAPIBackend(
            api_key=agent_config.api_key,
            model=agent_config.model or 'claude-sonnet-4-20250514',
            system_prompt=agent_config.system_prompt
        )

    elif backend_type == 'ollama-api':
        from backends import OllamaAPIBackend
        return OllamaAPIBackend(
            host=agent_config.host or 'localhost',
            port=agent_config.port or 11434,
            model=agent_config.model or 'llama3',
            system_prompt=agent_config.system_prompt,
            timeout=agent_config.timeout
        )

    elif backend_type == 'ollama-cli':
        from backends import OllamaCLIBackend
        return OllamaCLIBackend(
            command=agent_config.command or 'ollama',
            model=agent_config.model or 'llama3',
            timeout=agent_config.timeout
        )

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="UnifyWeaver Agent Loop - Terminal-friendly AI assistant"
    )

    # Agent selection (from config)
    parser.add_argument(
        '--agent', '-a',
        default=None,
        help='Agent variant from config file (e.g., yolo, claude-opus, ollama)'
    )
    parser.add_argument(
        '--config', '-C',
        default=None,
        help='Path to config file (agents.yaml or agents.json)'
    )
    parser.add_argument(
        '--list-agents',
        action='store_true',
        help='List available agent variants from config'
    )
    parser.add_argument(
        '--init-config',
        metavar='PATH',
        help='Create example config file at PATH'
    )

    # Direct backend selection (overrides config)
    parser.add_argument(
        '--backend', '-b',
        choices=['coro', 'claude', 'claude-code', 'gemini', 'ollama-api', 'ollama-cli'],
        default=None,
        help='Backend to use (overrides --agent config)'
    )
    parser.add_argument(
        '--command', '-c',
        default=None,
        help='Command for CLI backends'
    )
    parser.add_argument(
        '--model', '-m',
        default=None,
        help='Model to use'
    )
    parser.add_argument(
        '--host',
        default=None,
        help='Host for network backends (ollama-api)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='Port for network backends (ollama-api)'
    )
    parser.add_argument(
        '--api-key',
        help='API key for Claude API backend'
    )

    # Options
    parser.add_argument(
        '--no-tokens',
        action='store_true',
        help='Hide token usage information'
    )
    parser.add_argument(
        '--context-mode',
        choices=['continue', 'fresh', 'sliding'],
        default=None,
        help='Context behavior mode'
    )
    parser.add_argument(
        '--auto-tools',
        action='store_true',
        help='Auto-execute tools without confirmation'
    )
    parser.add_argument(
        '--no-tools',
        action='store_true',
        help='Disable tool execution'
    )
    parser.add_argument(
        '--system-prompt',
        default=None,
        help='System prompt to use'
    )
    parser.add_argument(
        '--max-iterations', '-i',
        type=int,
        default=None,
        help='Max tool iterations before pausing (0 = unlimited)'
    )

    # Prompt
    parser.add_argument(
        'prompt',
        nargs='?',
        help='Single prompt to run (non-interactive mode)'
    )

    args = parser.parse_args()

    # Handle --init-config
    if args.init_config:
        path = Path(args.init_config)
        save_example_config(path)
        print(f"Created example config at: {path}")
        return

    # Load configuration
    config = None
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config_from_dir()

    if config is None:
        config = get_default_config()

    # Handle --list-agents
    if args.list_agents:
        print("Available agents:")
        for name, agent in config.agents.items():
            default_marker = " (default)" if name == config.default else ""
            print(f"  {name}: {agent.backend} ({agent.model or 'default'}){default_marker}")
        return

    # Get agent config
    agent_name = args.agent or config.default
    if agent_name not in config.agents:
        print(f"Unknown agent: {agent_name}", file=sys.stderr)
        print(f"Available: {', '.join(config.agents.keys())}", file=sys.stderr)
        sys.exit(1)

    agent_config = config.agents[agent_name]

    # Override with command line args
    if args.backend:
        agent_config.backend = args.backend
    if args.command:
        agent_config.command = args.command
    if args.model:
        agent_config.model = args.model
    if args.host:
        agent_config.host = args.host
    if args.port:
        agent_config.port = args.port
    if args.api_key:
        agent_config.api_key = args.api_key
    if args.context_mode:
        agent_config.context_mode = args.context_mode
    if args.auto_tools:
        agent_config.auto_tools = True
    if args.no_tools:
        agent_config.tools = []
    if args.system_prompt:
        agent_config.system_prompt = args.system_prompt
    if args.max_iterations is not None:
        agent_config.max_iterations = args.max_iterations

    # Create backend
    try:
        backend = create_backend_from_config(agent_config)
    except ImportError as e:
        print(f"Backend requires additional package: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Create context manager
    behavior = ContextBehavior(agent_config.context_mode)
    context = ContextManager(
        behavior=behavior,
        max_tokens=agent_config.max_context_tokens,
        max_messages=agent_config.max_messages
    )

    # Create tool handler
    tools = None
    if agent_config.tools:
        tools = ToolHandler(
            allowed_tools=agent_config.tools,
            confirm_destructive=not agent_config.auto_tools
        )

    # Create and run loop
    loop = AgentLoop(
        backend=backend,
        context=context,
        tools=tools,
        show_tokens=agent_config.show_tokens and not args.no_tokens,
        auto_execute_tools=agent_config.auto_tools,
        max_iterations=agent_config.max_iterations,
        config=config
    )

    # Single prompt mode
    if args.prompt:
        print(f"You: {args.prompt}")
        loop._process_message(args.prompt)
        return

    # Interactive mode
    loop.run()


if __name__ == '__main__':
    main()
