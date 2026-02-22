"""Skills and agent.md loader for customizing agent behavior."""

from pathlib import Path
from typing import Any


class SkillsLoader:
    """Loads and combines skills and agent.md files into system prompts."""

    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    def load_agent_md(self, path: str | Path) -> str:
        """Load an agent.md file.

        The agent.md file defines the agent's persona, capabilities,
        and default instructions.
        """
        full_path = self._resolve_path(path)

        if not full_path.exists():
            raise FileNotFoundError(f"Agent file not found: {full_path}")

        return full_path.read_text()

    def load_skill(self, path: str | Path) -> str:
        """Load a single skill file.

        Skill files contain specific instructions for a capability
        (e.g., git operations, testing, code review).
        """
        full_path = self._resolve_path(path)

        if not full_path.exists():
            raise FileNotFoundError(f"Skill file not found: {full_path}")

        return full_path.read_text()

    def load_skills(self, paths: list[str | Path]) -> list[str]:
        """Load multiple skill files."""
        skills = []
        for path in paths:
            try:
                skills.append(self.load_skill(path))
            except FileNotFoundError as e:
                print(f"[Warning: {e}]")
        return skills

    def build_system_prompt(
        self,
        base_prompt: str | None = None,
        agent_md: str | Path | None = None,
        skills: list[str | Path] | None = None
    ) -> str:
        """Build a complete system prompt from components.

        Order:
        1. agent.md (defines persona and core instructions)
        2. Skills (specialized capabilities)
        3. Base prompt (additional instructions/overrides)
        """
        parts = []

        # Load agent.md if provided
        if agent_md:
            try:
                content = self.load_agent_md(agent_md)
                parts.append(content)
            except FileNotFoundError as e:
                print(f"[Warning: {e}]")

        # Load skills
        if skills:
            skill_contents = self.load_skills(skills)
            if skill_contents:
                parts.append("\n## Skills\n")
                for skill in skill_contents:
                    parts.append(skill)
                    parts.append("")  # Blank line separator

        # Add base prompt
        if base_prompt:
            if parts:
                parts.append("\n## Additional Instructions\n")
            parts.append(base_prompt)

        return "\n".join(parts) if parts else ""

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve a path relative to base_dir if not absolute."""
        path = Path(path)
        if path.is_absolute():
            return path
        return self.base_dir / path


def create_example_agent_md(path: str | Path):
    """Create an example agent.md file."""
    content = """# Coding Assistant

You are a skilled software engineer with expertise in multiple programming languages.

## Principles

- Write clean, readable, maintainable code
- Follow established conventions and patterns
- Test your changes when possible
- Explain your reasoning

## Communication Style

- Be concise and direct
- Use code examples when helpful
- Ask clarifying questions when requirements are unclear

## Tools

You have access to:
- `bash`: Execute shell commands
- `read`: Read file contents
- `write`: Create or overwrite files
- `edit`: Make targeted edits to files

## Process

1. Understand the request
2. Explore relevant code if needed
3. Plan the approach
4. Implement changes
5. Verify the changes work
"""
    Path(path).write_text(content)


def create_example_skill(path: str | Path, skill_type: str = "git"):
    """Create an example skill file."""
    skills = {
        "git": """## Git Skill

You are proficient with Git version control.

### Commit Guidelines
- Write clear, descriptive commit messages
- Use conventional commit format: type(scope): description
- Keep commits focused and atomic

### Common Operations
- Always check `git status` before committing
- Review diffs before staging changes
- Pull before pushing to avoid conflicts

### Branch Naming
- feature/description for new features
- fix/description for bug fixes
- docs/description for documentation
""",
        "testing": """## Testing Skill

You write thorough tests for code changes.

### Test Types
- Unit tests for individual functions
- Integration tests for component interactions
- End-to-end tests for user workflows

### Best Practices
- Test both happy path and edge cases
- Use descriptive test names
- Keep tests independent and isolated
- Mock external dependencies

### Commands
- Run tests with appropriate test runner
- Check test coverage when available
""",
        "code-review": """## Code Review Skill

You perform thorough code reviews.

### Review Checklist
- [ ] Code is readable and well-structured
- [ ] No obvious bugs or logic errors
- [ ] Error handling is appropriate
- [ ] No security vulnerabilities
- [ ] Tests cover the changes
- [ ] Documentation is updated

### Feedback Style
- Be constructive and specific
- Explain the "why" behind suggestions
- Distinguish between required changes and suggestions
"""
    }

    content = skills.get(skill_type, skills["git"])
    Path(path).write_text(content)