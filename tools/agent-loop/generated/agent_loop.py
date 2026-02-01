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


class AgentLoop:
    """Main agent loop orchestrating all components."""

    def __init__(
        self,
        backend: AgentBackend,
        context: ContextManager | None = None,
        show_tokens: bool = True
    ):
        self.backend = backend
        self.context = context or ContextManager()
        self.show_tokens = show_tokens
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
        'prompt',
        nargs='?',
        help='Single prompt to run (non-interactive mode)'
    )

    args = parser.parse_args()

    # Create backend
    if args.backend == 'coro':
        backend = CoroBackend(command=args.command)
    else:
        # TODO: Add Claude API backend
        print(f"Backend '{args.backend}' not yet implemented", file=sys.stderr)
        sys.exit(1)

    # Create context manager
    behavior = ContextBehavior(args.context_mode)
    context = ContextManager(behavior=behavior)

    # Create and run loop
    loop = AgentLoop(
        backend=backend,
        context=context,
        show_tokens=not args.no_tokens
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
