"""Prompt templates for reusable prompt snippets."""

import json
import re
from pathlib import Path
from dataclasses import dataclass


# Built-in templates
BUILTIN_TEMPLATES = {
    "explain": "Explain the following code in detail:\n\n```\n{code}\n```",
    "review": "Please review this code for bugs, security issues, and improvements:\n\n```\n{code}\n```",
    "refactor": "Refactor this code to be cleaner and more maintainable:\n\n```\n{code}\n```",
    "test": "Write unit tests for this code:\n\n```\n{code}\n```",
    "doc": "Add documentation comments to this code:\n\n```\n{code}\n```",
    "fix": "Fix the bug in this code:\n\n```\n{code}\n```\n\nError: {error}",
    "convert": "Convert this {from_lang} code to {to_lang}:\n\n```{from_lang}\n{code}\n```",
    "summarize": "Summarize the following in {length} sentences:\n\n{text}",
    "translate": "Translate the following to {language}:\n\n{text}",
    "simplify": "Simplify this explanation for a {audience}:\n\n{text}",
    "debug": "Help me debug this issue:\n\nCode:\n```\n{code}\n```\n\nExpected: {expected}\nActual: {actual}",
    "optimize": "Optimize this code for {goal}:\n\n```\n{code}\n```",
    "regex": "Create a regex pattern that matches: {description}",
    "sql": "Write a SQL query to: {description}",
    "bash": "Write a bash command to: {description}",
    "git": "What git commands should I use to: {description}",
}


@dataclass
class Template:
    """A prompt template."""
    name: str
    template: str
    description: str = ""
    variables: list[str] = None

    def __post_init__(self):
        if self.variables is None:
            # Extract variables from template
            self.variables = re.findall(r'\{(\w+)\}', self.template)

    def render(self, **kwargs) -> str:
        """Render the template with provided values."""
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def missing_variables(self, **kwargs) -> list[str]:
        """Return list of variables not provided."""
        return [v for v in self.variables if v not in kwargs]


class TemplateManager:
    """Manages prompt templates."""

    def __init__(self, config_path: str | Path | None = None):
        self.templates: dict[str, Template] = {}
        self.config_path = Path(config_path) if config_path else self._default_config_path()

        # Load built-in templates
        for name, tmpl in BUILTIN_TEMPLATES.items():
            self.templates[name] = Template(name=name, template=tmpl)

        # Load user templates
        self._load_user_templates()

    def _default_config_path(self) -> Path:
        return Path.home() / ".agent-loop" / "templates.json"

    def _load_user_templates(self) -> None:
        """Load user-defined templates from config file."""
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text())
                for name, tmpl_data in data.items():
                    if isinstance(tmpl_data, str):
                        self.templates[name] = Template(name=name, template=tmpl_data)
                    elif isinstance(tmpl_data, dict):
                        self.templates[name] = Template(
                            name=name,
                            template=tmpl_data.get("template", ""),
                            description=tmpl_data.get("description", "")
                        )
            except (json.JSONDecodeError, IOError):
                pass

    def save_templates(self) -> None:
        """Save user templates to config file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        # Only save non-builtin templates
        user_templates = {
            name: {"template": t.template, "description": t.description}
            for name, t in self.templates.items()
            if name not in BUILTIN_TEMPLATES
        }
        self.config_path.write_text(json.dumps(user_templates, indent=2))

    def get(self, name: str) -> Template | None:
        """Get a template by name."""
        return self.templates.get(name)

    def add(self, name: str, template: str, description: str = "") -> None:
        """Add or update a template."""
        self.templates[name] = Template(name=name, template=template, description=description)

    def remove(self, name: str) -> bool:
        """Remove a template. Returns True if removed."""
        if name in self.templates and name not in BUILTIN_TEMPLATES:
            del self.templates[name]
            return True
        return False

    def list_templates(self) -> list[Template]:
        """List all templates."""
        return sorted(self.templates.values(), key=lambda t: t.name)

    def render(self, name: str, **kwargs) -> str | None:
        """Render a template with provided values."""
        template = self.get(name)
        if template is None:
            return None
        return template.render(**kwargs)

    def format_list(self) -> str:
        """Format templates for display."""
        lines = ["Templates:"]

        # Built-in
        lines.append("\n  Built-in:")
        for name in sorted(BUILTIN_TEMPLATES.keys()):
            t = self.templates[name]
            vars_str = ", ".join(t.variables) if t.variables else "none"
            lines.append(f"    @{name} ({vars_str})")

        # User-defined
        user_templates = [t for t in self.templates.values() if t.name not in BUILTIN_TEMPLATES]
        if user_templates:
            lines.append("\n  User-defined:")
            for t in sorted(user_templates, key=lambda x: x.name):
                vars_str = ", ".join(t.variables) if t.variables else "none"
                lines.append(f"    @{t.name} ({vars_str})")

        return "\n".join(lines)

    def parse_template_invocation(self, text: str) -> tuple[str | None, dict]:
        """Parse a template invocation like '@explain code=...'

        Returns (template_name, kwargs) or (None, {}) if not a template.
        """
        if not text.startswith('@'):
            return None, {}

        parts = text[1:].split(None, 1)
        if not parts:
            return None, {}

        name = parts[0]
        if name not in self.templates:
            return None, {}

        kwargs = {}
        if len(parts) > 1:
            # Parse key=value pairs or use remaining as first variable
            rest = parts[1]
            template = self.templates[name]

            # Check for key=value format
            kv_pattern = r'(\w+)=(?:"([^"]*)"|(\S+))'
            matches = re.findall(kv_pattern, rest)

            if matches:
                for key, quoted, unquoted in matches:
                    kwargs[key] = quoted if quoted else unquoted
            elif template.variables:
                # Use rest as the first variable
                kwargs[template.variables[0]] = rest

        return name, kwargs


def create_default_templates_file(path: str | Path | None = None) -> Path:
    """Create a default templates file for user customization."""
    path = Path(path) if path else Path.home() / ".agent-loop" / "templates.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    example = {
        "mytemplate": {
            "template": "Do something with {input}",
            "description": "Example custom template"
        },
        "pr": {
            "template": "Create a pull request description for these changes:\n\n{changes}",
            "description": "Generate PR description"
        }
    }

    path.write_text(json.dumps(example, indent=2))
    return path