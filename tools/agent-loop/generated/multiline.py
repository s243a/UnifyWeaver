"""Multi-line input support for the agent loop."""

import sys


def get_multiline_input(prompt: str = "You: ", end_marker: str = "EOF") -> str | None:
    """Get multi-line input from the user.

    Input ends when:
    - User enters the end_marker on its own line (default: EOF)
    - User presses Ctrl+D (EOF)
    - User enters a blank line after text (optional quick mode)

    Returns None on true EOF (Ctrl+D with no input).
    """
    print(f"{prompt}(multi-line mode, enter '{end_marker}' or Ctrl+D to finish)")
    lines = []

    try:
        while True:
            line = input()
            if line.strip() == end_marker:
                break
            lines.append(line)
    except EOFError:
        if not lines:
            return None

    return "\n".join(lines)


def get_input_smart(prompt: str = "You: ") -> str | None:
    """Smart input that detects when multi-line is needed.

    Triggers multi-line mode when:
    - Input starts with ``` (code block)
    - Input starts with <<< (heredoc style)
    - Input ends with \\ (line continuation)
    - Input is just '{' or '[' (JSON/data structure)

    Returns None on EOF.
    """
    try:
        line = input(prompt)
    except EOFError:
        return None

    if not line:
        return ""

    stripped = line.strip()

    # Check for multi-line triggers
    if stripped.startswith("```"):
        # Code block mode - read until closing ```
        lines = [line]
        try:
            while True:
                next_line = input()
                lines.append(next_line)
                if next_line.strip() == "```" or next_line.strip().startswith("```"):
                    break
        except EOFError:
            pass
        return "\n".join(lines)

    if stripped.startswith("<<<"):
        # Heredoc mode - read until marker
        marker = stripped[3:].strip() or "EOF"
        lines = []
        print(f"(enter '{marker}' to finish)")
        try:
            while True:
                next_line = input()
                if next_line.strip() == marker:
                    break
                lines.append(next_line)
        except EOFError:
            pass
        return "\n".join(lines)

    if stripped.endswith("\\"):
        # Line continuation mode
        lines = [line.rstrip("\\")]
        try:
            while True:
                next_line = input("... ")
                if next_line.endswith("\\"):
                    lines.append(next_line.rstrip("\\"))
                else:
                    lines.append(next_line)
                    break
        except EOFError:
            pass
        return "\n".join(lines)

    if stripped in ('{', '[', '('):
        # Data structure mode - read until matching close
        open_char = stripped
        close_char = {'{': '}', '[': ']', '(': ')'}[open_char]
        lines = [line]
        depth = 1
        print(f"(enter until matching '{close_char}')")
        try:
            while depth > 0:
                next_line = input("... ")
                lines.append(next_line)
                # Simple depth tracking (doesn't handle strings properly)
                depth += next_line.count(open_char) - next_line.count(close_char)
        except EOFError:
            pass
        return "\n".join(lines)

    return line


class MultilineInputHandler:
    """Handles multi-line input with various modes."""

    def __init__(
        self,
        prompt: str = "You: ",
        continuation_prompt: str = "... ",
        end_marker: str = "EOF"
    ):
        self.prompt = prompt
        self.continuation_prompt = continuation_prompt
        self.end_marker = end_marker
        self.multiline_mode = False

    def get_input(self) -> str | None:
        """Get input, handling multi-line when needed."""
        if self.multiline_mode:
            return self._get_explicit_multiline()
        return get_input_smart(self.prompt)

    def _get_explicit_multiline(self) -> str | None:
        """Get multi-line input in explicit mode."""
        return get_multiline_input(self.prompt, self.end_marker)

    def toggle_multiline(self) -> bool:
        """Toggle explicit multi-line mode. Returns new state."""
        self.multiline_mode = not self.multiline_mode
        return self.multiline_mode


def paste_mode() -> str:
    """Enter paste mode for pasting large blocks of text.

    Reads all input until Ctrl+D without any processing.
    """
    print("Paste mode: paste your text, then press Ctrl+D")
    lines = []
    try:
        while True:
            line = sys.stdin.readline()
            if not line:  # EOF
                break
            lines.append(line.rstrip('\n'))
    except KeyboardInterrupt:
        pass

    return "\n".join(lines)
