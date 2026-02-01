#!/usr/bin/env python3
"""
UnifyWeaver Agent Loop - Append-Only Mode

A simple agent loop that works in any terminal environment.
Uses pure text output with no cursor control or escape codes.
"""

import sys
import argparse
from backends import AgentBackend, CoroBackend
from context import ContextManager, ContextBehavior
from tools import ToolHandler


class AgentLoop:
    """Main agent loop orchestrating all components."""

    def __init__(
        self,
        backend: AgentBackend,
        context: ContextManager | None = None,
        tools: ToolHandler | None = None,
        show_tokens: bool = True,
        auto_execute_tools: bool = False
    ):
        self.backend = backend
        self.context = context or ContextManager()
        self.tools = tools or ToolHandler()
        self.show_tokens = show_tokens
        self.auto_execute_tools = auto_execute_tools
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
        print("Type 'exit' to quit, 'clear' to reset context, 'help' for commands")
        print()

    def _get_input(self) -> str | None:
        """Get input from user."""
        try:
            return input("You: ")
        except EOFError:
            return None

    def _handle_command(self, user_input: str) -> bool:
        """Handle special commands. Returns True if command was handled."""
        cmd = user_input.strip().lower()

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

        return False

    def _print_help(self) -> None:
        """Print help message."""
        print("""
Commands:
  exit, quit  - Exit the agent loop
  clear       - Clear conversation context
  status      - Show context status (messages, tokens)
  help        - Show this help message

Just type your message to chat with the agent.
""")

    def _print_status(self) -> None:
        """Print context status."""
        print(f"""
Context Status:
  Messages: {self.context.message_count}
  Tokens: {self.context.token_count}
  Backend: {self.backend.name}
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
        while response.tool_calls:
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

            print("\n[Continuing with tool results...]")
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


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="UnifyWeaver Agent Loop - Terminal-friendly AI assistant"
    )
    parser.add_argument(
        '--backend', '-b',
        choices=['coro', 'claude'],
        default='coro',
        help='Backend to use (default: coro)'
    )
    parser.add_argument(
        '--command', '-c',
        default='claude',
        help='Command for coro backend (default: claude)'
    )
    parser.add_argument(
        '--no-tokens',
        action='store_true',
        help='Hide token usage information'
    )
    parser.add_argument(
        '--context-mode',
        choices=['continue', 'fresh', 'sliding'],
        default='continue',
        help='Context behavior mode (default: continue)'
    )
    parser.add_argument(
        '--api-key',
        help='API key for Claude backend (or set ANTHROPIC_API_KEY env var)'
    )
    parser.add_argument(
        '--model',
        default='claude-sonnet-4-20250514',
        help='Model for Claude API backend (default: claude-sonnet-4-20250514)'
    )
    parser.add_argument(
        '--auto-tools',
        action='store_true',
        help='Auto-execute tools without confirmation (use with caution)'
    )
    parser.add_argument(
        '--no-tools',
        action='store_true',
        help='Disable tool execution entirely'
    )
    parser.add_argument(
        'prompt',
        nargs='?',
        help='Single prompt to run (non-interactive mode)'
    )

    args = parser.parse_args()

    # Create backend
    if args.backend == 'coro':
        backend = CoroBackend(command=args.command)
    elif args.backend == 'claude':
        try:
            from backends import ClaudeAPIBackend
            backend = ClaudeAPIBackend(
                api_key=args.api_key,
                model=args.model
            )
        except ImportError as e:
            print(f"Claude API backend requires anthropic package: {e}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Unknown backend: {args.backend}", file=sys.stderr)
        sys.exit(1)

    # Create context manager
    behavior = ContextBehavior(args.context_mode)
    context = ContextManager(behavior=behavior)

    # Create tool handler
    tools = None if args.no_tools else ToolHandler(
        confirm_destructive=not args.auto_tools
    )

    # Create and run loop
    loop = AgentLoop(
        backend=backend,
        context=context,
        tools=tools,
        show_tokens=not args.no_tokens,
        auto_execute_tools=args.auto_tools
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
