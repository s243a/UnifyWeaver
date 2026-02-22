"""Command aliases for the agent loop."""

import json
from pathlib import Path
from typing import Callable


# Default aliases
DEFAULT_ALIASES = {
    # Short forms
    "q": "quit",
    "x": "exit",
    "h": "help",
    "?": "help",
    "c": "clear",
    "s": "status",

    # Save/load shortcuts
    "sv": "save",
    "ld": "load",
    "ls": "sessions",

    # Export shortcuts
    "exp": "export",
    "md": "export conversation.md",
    "html": "export conversation.html",

    # Backend shortcuts
    "be": "backend",
    "sw": "backend",  # switch

    # Common backend switches
    "yolo": "backend yolo",
    "opus": "backend claude-opus",
    "sonnet": "backend claude-sonnet",
    "haiku": "backend claude-haiku",
    "gpt": "backend openai",
    "local": "backend ollama",

    # Format shortcuts
    "fmt": "format",

    # Iteration shortcuts
    "iter": "iterations",
    "i0": "iterations 0",
    "i1": "iterations 1",
    "i3": "iterations 3",
    "i5": "iterations 5",

    # Stream toggle
    "str": "stream",

    # Cost
    "$": "cost",

    # Search
    "find": "search",
    "grep": "search",
}


class AliasManager:
    """Manages command aliases."""

    def __init__(self, config_path: str | Path | None = None):
        self.aliases = DEFAULT_ALIASES.copy()
        self.config_path = Path(config_path) if config_path else self._default_config_path()
        self._load_user_aliases()

    def _default_config_path(self) -> Path:
        return Path.home() / ".agent-loop" / "aliases.json"

    def _load_user_aliases(self) -> None:
        """Load user-defined aliases from config file."""
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text())
                self.aliases.update(data)
            except (json.JSONDecodeError, IOError):
                pass

    def save_aliases(self) -> None:
        """Save current aliases to config file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        # Only save non-default aliases
        user_aliases = {
            k: v for k, v in self.aliases.items()
            if k not in DEFAULT_ALIASES or DEFAULT_ALIASES.get(k) != v
        }
        self.config_path.write_text(json.dumps(user_aliases, indent=2))

    def resolve(self, command: str) -> str:
        """Resolve an alias to its full command.

        Returns the original command if no alias found.
        """
        # Handle slash prefix
        if command.startswith('/'):
            cmd = command[1:]
            prefix = '/'
        else:
            cmd = command
            prefix = ''

        # Check for exact alias match (first word only)
        parts = cmd.split(None, 1)
        if parts:
            first_word = parts[0].lower()
            if first_word in self.aliases:
                # Replace first word with alias expansion
                expanded = self.aliases[first_word]
                if len(parts) > 1:
                    # Append any additional arguments
                    expanded = f"{expanded} {parts[1]}"
                return prefix + expanded

        return command

    def add(self, alias: str, command: str) -> None:
        """Add or update an alias."""
        self.aliases[alias.lower()] = command

    def remove(self, alias: str) -> bool:
        """Remove an alias. Returns True if removed."""
        alias = alias.lower()
        if alias in self.aliases:
            del self.aliases[alias]
            return True
        return False

    def list_aliases(self) -> list[tuple[str, str]]:
        """List all aliases sorted by name."""
        return sorted(self.aliases.items())

    def format_list(self) -> str:
        """Format aliases for display."""
        lines = ["Aliases:"]

        # Group by category
        categories = {
            "Navigation": ["q", "x", "h", "?", "c", "s"],
            "Sessions": ["sv", "ld", "ls"],
            "Export": ["exp", "md", "html"],
            "Backend": ["be", "sw", "yolo", "opus", "sonnet", "haiku", "gpt", "local"],
            "Iterations": ["iter", "i0", "i1", "i3", "i5"],
            "Other": ["fmt", "str", "$", "find", "grep"],
        }

        for category, keys in categories.items():
            cat_aliases = [(k, self.aliases[k]) for k in keys if k in self.aliases]
            if cat_aliases:
                lines.append(f"\n  {category}:")
                for alias, cmd in cat_aliases:
                    lines.append(f"    /{alias} -> /{cmd}")

        # User-defined aliases
        user_aliases = [
            (k, v) for k, v in self.aliases.items()
            if k not in DEFAULT_ALIASES
        ]
        if user_aliases:
            lines.append("\n  User-defined:")
            for alias, cmd in sorted(user_aliases):
                lines.append(f"    /{alias} -> /{cmd}")

        return "\n".join(lines)


def create_default_aliases_file(path: str | Path | None = None) -> Path:
    """Create a default aliases file for user customization."""
    path = Path(path) if path else Path.home() / ".agent-loop" / "aliases.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    example = {
        "# Comment": "Add your custom aliases here",
        "myalias": "backend claude-opus",
        "quick": "backend claude-haiku",
    }
    # Remove comment key
    del example["# Comment"]

    path.write_text(json.dumps(example, indent=2))
    return path
