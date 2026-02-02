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
from context import ContextManager, ContextBehavior, ContextFormat
from tools import ToolHandler
from config import load_config, load_config_from_dir, get_default_config, AgentConfig, Config, save_example_config
from sessions import SessionManager
from skills import SkillsLoader
from costs import CostTracker
from export import ConversationExporter
from search import SessionSearcher
from aliases import AliasManager
from templates import TemplateManager
from history import HistoryManager
from multiline import get_input_smart


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
        config: Config | None = None,
        session_manager: SessionManager | None = None,
        session_id: str | None = None,
        track_costs: bool = True,
        streaming: bool = False
    ):
        self.backend = backend
        self.context = context or ContextManager()
        self.tools = tools or ToolHandler()
        self.show_tokens = show_tokens
        self.auto_execute_tools = auto_execute_tools
        self.max_iterations = max_iterations  # 0 = unlimited
        self.config = config  # For backend switching
        self.session_manager = session_manager or SessionManager()
        self.session_id = session_id  # Current session ID if loaded/saved
        self.cost_tracker = CostTracker() if track_costs else None
        self.streaming = streaming
        self.alias_manager = AliasManager()
        self.template_manager = TemplateManager()
        self.history_manager = HistoryManager(self.context)
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

                # Handle template invocation (@template_name ...)
                if user_input.strip().startswith('@'):
                    name, kwargs = self.template_manager.parse_template_invocation(user_input.strip())
                    if name:
                        template = self.template_manager.get(name)
                        missing = template.missing_variables(**kwargs)
                        if missing:
                            print(f"[Template @{name} requires: {', '.join(missing)}]\n")
                            continue
                        user_input = template.render(**kwargs)
                        print(f"[Using template @{name}]")

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
        """Get input from user with multi-line support."""
        return get_input_smart("You: ")

    def _handle_command(self, user_input: str) -> bool:
        """Handle special commands. Returns True if command was handled."""
        # Apply alias resolution first
        resolved = self.alias_manager.resolve(user_input)
        if resolved != user_input:
            user_input = resolved

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
            self.history_manager = HistoryManager(self.context)
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

        # Session commands
        if cmd.startswith('save'):
            return self._handle_save_command(text)

        if cmd.startswith('load'):
            return self._handle_load_command(text)

        if cmd == 'sessions':
            return self._handle_sessions_command()

        # /format [type] - set context format
        if cmd.startswith('format'):
            return self._handle_format_command(text)

        # /export <path> - export conversation
        if cmd.startswith('export'):
            return self._handle_export_command(text)

        # /cost - show cost tracking
        if cmd == 'cost' or cmd == 'costs':
            return self._handle_cost_command()

        # /search <query> - search sessions
        if cmd.startswith('search'):
            return self._handle_search_command(text)

        # /stream - toggle streaming mode
        if cmd == 'stream' or cmd == 'streaming':
            return self._handle_stream_command()

        # Aliases
        if cmd == 'aliases':
            return self._handle_aliases_command()

        # Templates
        if cmd == 'templates':
            return self._handle_templates_command()

        # History
        if cmd == 'history' or cmd.startswith('history '):
            return self._handle_history_command(text)

        # Undo
        if cmd == 'undo':
            return self._handle_undo_command()

        # Delete message(s)
        if cmd.startswith('delete ') or cmd.startswith('del '):
            return self._handle_delete_command(text)

        # Edit message
        if cmd.startswith('edit '):
            return self._handle_edit_command(text)

        # Replay from message
        if cmd.startswith('replay '):
            return self._handle_replay_command(text)

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

    def _handle_save_command(self, text: str) -> bool:
        """Handle /save [name] command."""
        parts = text.split(None, 1)
        name = parts[1].strip() if len(parts) > 1 else None

        try:
            self.session_id = self.session_manager.save_session(
                context=self.context,
                session_id=self.session_id,
                name=name,
                backend_name=self.backend.name
            )
            print(f"[Session saved: {self.session_id}]\n")
        except Exception as e:
            print(f"[Error saving session: {e}]\n")
        return True

    def _handle_load_command(self, text: str) -> bool:
        """Handle /load <session_id> command."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            print("[Usage: /load <session_id>]")
            print("[Use /sessions to list available sessions]\n")
            return True

        session_id = parts[1].strip()

        try:
            context, metadata, extra = self.session_manager.load_session(session_id)
            self.context = context
            self.session_id = session_id
            print(f"[Session loaded: {metadata.get('name', session_id)}]")
            print(f"[Messages: {context.message_count}, Tokens: {context.token_count}]\n")
        except FileNotFoundError:
            print(f"[Session not found: {session_id}]\n")
        except Exception as e:
            print(f"[Error loading session: {e}]\n")
        return True

    def _handle_sessions_command(self) -> bool:
        """Handle /sessions command."""
        sessions = self.session_manager.list_sessions()
        if not sessions:
            print("[No saved sessions]\n")
            return True

        print("Saved sessions:")
        for s in sessions[:10]:  # Show last 10
            print(f"  {s.id}: {s.name} ({s.message_count} msgs, {s.backend})")
        if len(sessions) > 10:
            print(f"  ... and {len(sessions) - 10} more")
        print()
        return True

    def _handle_format_command(self, text: str) -> bool:
        """Handle /format [type] command."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            print(f"[Current format: {self.context.format.value}]")
            print("[Available: plain, markdown, json, xml]\n")
            return True

        format_name = parts[1].strip().lower()
        try:
            self.context.format = ContextFormat(format_name)
            print(f"[Context format set to: {format_name}]\n")
        except ValueError:
            print(f"[Unknown format: {format_name}]")
            print("[Available: plain, markdown, json, xml]\n")
        return True

    def _handle_export_command(self, text: str) -> bool:
        """Handle /export <path> command."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            print("[Usage: /export <path>]")
            print("[Formats: .md, .html, .json, .txt (auto-detected from extension)]\n")
            return True

        path = parts[1].strip()
        try:
            exporter = ConversationExporter(self.context)
            exporter.save(path, title=self.session_id or "Conversation")
            print(f"[Exported to: {path}]\n")
        except Exception as e:
            print(f"[Error exporting: {e}]\n")
        return True

    def _handle_cost_command(self) -> bool:
        """Handle /cost command."""
        if not self.cost_tracker:
            print("[Cost tracking disabled]\n")
            return True

        summary = self.cost_tracker.get_summary()
        print(f"""
Cost Summary:
  Requests: {summary['total_requests']}
  Input tokens: {summary['total_input_tokens']:,}
  Output tokens: {summary['total_output_tokens']:,}
  Total tokens: {summary['total_tokens']:,}
  Estimated cost: {summary['cost_formatted']}
""")
        return True

    def _handle_search_command(self, text: str) -> bool:
        """Handle /search <query> command."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            print("[Usage: /search <query>]\n")
            return True

        query = parts[1].strip()
        try:
            searcher = SessionSearcher(self.session_manager.sessions_dir)
            results = searcher.search(query, limit=10)

            if not results:
                print(f"[No results for: {query}]\n")
                return True

            print(f"Search results for '{query}':")
            for r in results:
                role = "You" if r.role == "user" else "Asst"
                snippet = r.content[r.match_start:r.match_end]
                context = r.context_before[-20:] + "**" + snippet + "**" + r.context_after[:20]
                print(f"  [{r.session_id}] {role}: ...{context}...")
            print()
        except Exception as e:
            print(f"[Search error: {e}]\n")
        return True

    def _handle_stream_command(self) -> bool:
        """Handle /stream command to toggle streaming mode."""
        if not self.backend.supports_streaming():
            print(f"[Backend {self.backend.name} does not support streaming]\n")
            return True

        self.streaming = not self.streaming
        status = "enabled" if self.streaming else "disabled"
        print(f"[Streaming {status}]\n")
        return True

    def _handle_aliases_command(self) -> bool:
        """Handle /aliases command."""
        print(self.alias_manager.format_list())
        print()
        return True

    def _handle_templates_command(self) -> bool:
        """Handle /templates command."""
        print(self.template_manager.format_list())
        print()
        return True

    def _handle_history_command(self, text: str) -> bool:
        """Handle /history [n] command."""
        parts = text.split()
        limit = 10
        if len(parts) > 1:
            try:
                limit = int(parts[1])
            except ValueError:
                pass
        print(self.history_manager.format_history(limit))
        print()
        return True

    def _handle_undo_command(self) -> bool:
        """Handle /undo command."""
        if self.history_manager.undo():
            print("[Undo successful]\n")
        else:
            print("[Nothing to undo]\n")
        return True

    def _handle_delete_command(self, text: str) -> bool:
        """Handle /delete <n> or /delete <start>-<end> command."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            print("[Usage: /delete <index> or /delete <start>-<end> or /delete last [n]]\n")
            return True

        arg = parts[1].strip()

        # Handle 'last [n]'
        if arg.startswith('last'):
            n = 1
            sub_parts = arg.split()
            if len(sub_parts) > 1:
                try:
                    n = int(sub_parts[1])
                except ValueError:
                    pass
            count = self.history_manager.delete_last(n)
            print(f"[Deleted {count} message(s)]\n")
            return True

        # Handle range 'start-end'
        if '-' in arg:
            try:
                start, end = arg.split('-', 1)
                start = int(start)
                end = int(end)
                count = self.history_manager.delete_range(start, end)
                print(f"[Deleted {count} message(s)]\n")
            except ValueError:
                print("[Invalid range format]\n")
            return True

        # Handle single index
        try:
            index = int(arg)
            if self.history_manager.delete_message(index):
                print(f"[Deleted message {index}]\n")
            else:
                print(f"[Invalid index: {index}]\n")
        except ValueError:
            print("[Invalid index]\n")
        return True

    def _handle_edit_command(self, text: str) -> bool:
        """Handle /edit <index> command."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            print("[Usage: /edit <index>]\n")
            return True

        try:
            index = int(parts[1].strip())
        except ValueError:
            print("[Invalid index]\n")
            return True

        # Get current content
        content = self.history_manager.get_full_content(index)
        if content is None:
            print(f"[Invalid index: {index}]\n")
            return True

        print(f"Current content of message {index}:")
        print(content[:200] + "..." if len(content) > 200 else content)
        print("\nEnter new content (or empty to cancel):")

        try:
            new_content = get_input_smart("New: ")
            if new_content and new_content.strip():
                if self.history_manager.edit_message(index, new_content):
                    print(f"[Message {index} updated]\n")
                else:
                    print("[Edit failed]\n")
            else:
                print("[Edit cancelled]\n")
        except EOFError:
            print("[Edit cancelled]\n")
        return True

    def _handle_replay_command(self, text: str) -> bool:
        """Handle /replay <index> command to re-send a message."""
        parts = text.split(None, 1)
        if len(parts) < 2:
            print("[Usage: /replay <index>]\n")
            return True

        try:
            index = int(parts[1].strip())
        except ValueError:
            print("[Invalid index]\n")
            return True

        content = self.history_manager.replay_from(index)
        if content is None:
            print(f"[Cannot replay: message {index} not found or not a user message]\n")
            return True

        print(f"[Replaying message {index}]")
        self._process_message(content)
        return True

    def _print_help(self) -> None:
        """Print help message."""
        print("""
Commands (with or without / prefix):
  /exit, /quit       - Exit the agent loop
  /clear             - Clear conversation context
  /status            - Show context status
  /help              - Show this help message

Loop Control:
  /iterations [N]    - Show or set max iterations (0 = unlimited)
  /backend [name]    - Show or switch backend/agent
  /format [type]     - Set context format (plain/markdown/json/xml)
  /stream            - Toggle streaming mode for API backends

Sessions:
  /save [name]       - Save current conversation
  /load <id>         - Load a saved conversation
  /sessions          - List saved sessions
  /search <query>    - Search across saved sessions

Export & Costs:
  /export <path>     - Export conversation (.md, .html, .json, .txt)
  /cost              - Show API cost tracking summary

History:
  /history [n]       - Show last n messages (default 10)
  /delete <idx>      - Delete message at index
  /delete <s>-<e>    - Delete messages from s to e
  /delete last [n]   - Delete last n messages
  /edit <idx>        - Edit message at index
  /replay <idx>      - Re-send message at index
  /undo              - Undo last history change

Shortcuts:
  /aliases           - List command aliases (e.g., /q -> /quit)
  /templates         - List prompt templates

Multi-line Input:
  Start with ``` for code blocks
  Start with <<< for heredoc mode
  End lines with \\ for continuation

Just type your message to chat with the agent.
""")

    def _print_status(self) -> None:
        """Print context status."""
        iterations_str = "unlimited" if self.max_iterations == 0 else str(self.max_iterations)
        session_str = self.session_id or "(unsaved)"
        streaming_str = "on" if self.streaming else "off"
        cost_str = self.cost_tracker.format_status() if self.cost_tracker else "disabled"
        print(f"""
Status:
  Backend: {self.backend.name}
  Session: {session_str}
  Streaming: {streaming_str}
  Max iterations: {iterations_str}
  Context format: {self.context.format.value}
  Messages: {self.context.message_count}
  Context tokens: {self.context.token_count}
  Cost: {cost_str}
""")

    def _process_message(self, user_input: str) -> None:
        """Process a user message and get response."""
        # Add to context
        self.context.add_message('user', user_input)

        # Show that we're calling the backend
        print("\n[Calling backend...]")

        # Get response from backend (with streaming if enabled)
        if self.streaming and self.backend.supports_streaming() and hasattr(self.backend, 'send_message_streaming'):
            print("\nAssistant: ", end="", flush=True)
            response = self.backend.send_message_streaming(
                user_input,
                self.context.get_context(),
                on_token=lambda t: print(t, end="", flush=True)
            )
            print()  # Newline after streaming
        else:
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

        # Display response (if not already streamed)
        if not (self.streaming and self.backend.supports_streaming() and hasattr(self.backend, 'send_message_streaming')):
            print(f"\nAssistant: {response.content}\n")
        else:
            print()  # Just add spacing after streamed output

        # Add response to context
        output_tokens = response.tokens.get('output', 0)
        input_tokens = response.tokens.get('input', 0)
        self.context.add_message('assistant', response.content, output_tokens)

        # Track costs
        if self.cost_tracker and response.tokens:
            model = getattr(self.backend, 'model', 'unknown')
            self.cost_tracker.record_usage(model, input_tokens, output_tokens)

        # Show token summary if enabled
        if self.show_tokens and response.tokens:
            token_info = ", ".join(
                f"{k}: {v}" for k, v in response.tokens.items()
            )
            cost_info = ""
            if self.cost_tracker:
                cost_info = f" | Est. cost: {self.cost_tracker.get_summary()['cost_formatted']}"
            print(f"  [Tokens: {token_info}{cost_info}]\n")


def build_system_prompt(agent_config: AgentConfig, config_dir: str = "") -> str | None:
    """Build system prompt from config, agent.md, and skills."""
    loader = SkillsLoader(base_dir=config_dir or Path.cwd())

    return loader.build_system_prompt(
        base_prompt=agent_config.system_prompt,
        agent_md=agent_config.agent_md,
        skills=agent_config.skills if agent_config.skills else None
    ) or agent_config.system_prompt


def create_backend_from_config(agent_config: AgentConfig, config_dir: str = "") -> AgentBackend:
    """Create a backend from an AgentConfig."""
    backend_type = agent_config.backend

    # Build system prompt with skills/agent.md
    system_prompt = build_system_prompt(agent_config, config_dir)

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
            model=agent_config.model or 'gemini-3-flash-preview'
        )

    elif backend_type == 'claude':
        from backends import ClaudeAPIBackend
        return ClaudeAPIBackend(
            api_key=agent_config.api_key,
            model=agent_config.model or 'claude-sonnet-4-20250514',
            system_prompt=system_prompt
        )

    elif backend_type == 'openai':
        from backends import OpenAIBackend
        return OpenAIBackend(
            api_key=agent_config.api_key,
            model=agent_config.model or 'gpt-4o',
            system_prompt=system_prompt,
            base_url=agent_config.extra.get('base_url')
        )

    elif backend_type == 'ollama-api':
        from backends import OllamaAPIBackend
        return OllamaAPIBackend(
            host=agent_config.host or 'localhost',
            port=agent_config.port or 11434,
            model=agent_config.model or 'llama3',
            system_prompt=system_prompt,
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
        choices=['coro', 'claude', 'claude-code', 'gemini', 'openai', 'ollama-api', 'ollama-cli'],
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

    # Session management
    parser.add_argument(
        '--session', '-s',
        default=None,
        help='Load a saved session by ID'
    )
    parser.add_argument(
        '--list-sessions',
        action='store_true',
        help='List saved sessions'
    )
    parser.add_argument(
        '--sessions-dir',
        default=None,
        help='Directory for session files (default: ~/.agent-loop/sessions)'
    )

    # Context format
    parser.add_argument(
        '--context-format',
        choices=['plain', 'markdown', 'json', 'xml'],
        default=None,
        help='Format for context when sent to backend'
    )

    # Streaming and cost tracking
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Enable streaming output for API backends'
    )
    parser.add_argument(
        '--no-cost-tracking',
        action='store_true',
        help='Disable cost tracking'
    )

    # Search
    parser.add_argument(
        '--search',
        metavar='QUERY',
        help='Search across saved sessions and exit'
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

    # Create session manager
    session_manager = SessionManager(args.sessions_dir)

    # Handle --list-sessions
    if args.list_sessions:
        sessions = session_manager.list_sessions()
        if not sessions:
            print("No saved sessions")
            return
        print("Saved sessions:")
        for s in sessions:
            print(f"  {s.id}: {s.name} ({s.message_count} msgs, {s.backend})")
        return

    # Handle --search
    if args.search:
        searcher = SessionSearcher(session_manager.sessions_dir)
        results = searcher.search(args.search, limit=20)
        if not results:
            print(f"No results for: {args.search}")
            return
        print(f"Search results for '{args.search}':")
        for r in results:
            role = "You" if r.role == "user" else "Asst"
            snippet = r.content[r.match_start:r.match_end]
            context = r.context_before[-20:] + "**" + snippet + "**" + r.context_after[:20]
            print(f"  [{r.session_id}] {role}: ...{context}...")
        return

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
        # Clear command so backend uses its own default
        if not args.command:
            agent_config.command = None
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
        backend = create_backend_from_config(agent_config, config.config_dir)
    except ImportError as e:
        print(f"Backend requires additional package: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load session if specified
    session_id = None
    context = None

    if args.session:
        try:
            context, metadata, extra = session_manager.load_session(args.session)
            session_id = args.session
            print(f"Loaded session: {metadata.get('name', session_id)}")
        except FileNotFoundError:
            print(f"Session not found: {args.session}", file=sys.stderr)
            sys.exit(1)

    # Create context manager if not loaded from session
    if context is None:
        behavior = ContextBehavior(agent_config.context_mode)
        context_format = ContextFormat(args.context_format) if args.context_format else ContextFormat.PLAIN
        context = ContextManager(
            behavior=behavior,
            max_tokens=agent_config.max_context_tokens,
            max_messages=agent_config.max_messages,
            format=context_format
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
        config=config,
        session_manager=session_manager,
        session_id=session_id,
        track_costs=not args.no_cost_tracking,
        streaming=args.stream
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
