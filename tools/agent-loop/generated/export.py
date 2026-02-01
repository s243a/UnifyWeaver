"""Export conversations to various formats."""

from pathlib import Path
from datetime import datetime
from context import ContextManager


class ConversationExporter:
    """Export conversation history to different formats."""

    def __init__(self, context: ContextManager):
        self.context = context

    def to_markdown(self, title: str = "Conversation") -> str:
        """Export conversation to Markdown format."""
        lines = [
            f"# {title}",
            "",
            f"*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            f"*Messages: {self.context.message_count} | Tokens: {self.context.token_count}*",
            "",
            "---",
            ""
        ]

        for msg in self.context.messages:
            role = "**You:**" if msg.role == "user" else "**Assistant:**"
            lines.append(role)
            lines.append("")
            lines.append(msg.content)
            lines.append("")

        return "\n".join(lines)

    def to_html(self, title: str = "Conversation") -> str:
        """Export conversation to HTML format."""
        messages_html = []

        for msg in self.context.messages:
            role_class = "user" if msg.role == "user" else "assistant"
            role_label = "You" if msg.role == "user" else "Assistant"
            # Escape HTML and preserve newlines
            content = self._escape_html(msg.content).replace("\n", "<br>")
            messages_html.append(f'''
        <div class="message {role_class}">
            <div class="role">{role_label}</div>
            <div class="content">{content}</div>
        </div>''')

        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self._escape_html(title)}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
        }}
        .meta {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        .message {{
            margin: 15px 0;
            padding: 15px;
            border-radius: 8px;
        }}
        .message.user {{
            background: #e3f2fd;
            margin-left: 20px;
        }}
        .message.assistant {{
            background: #fff;
            border: 1px solid #ddd;
            margin-right: 20px;
        }}
        .role {{
            font-weight: bold;
            margin-bottom: 8px;
            color: #555;
        }}
        .content {{
            line-height: 1.6;
            white-space: pre-wrap;
        }}
        code {{
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
        }}
        pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <h1>{self._escape_html(title)}</h1>
    <div class="meta">
        Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
        Messages: {self.context.message_count} |
        Tokens: {self.context.token_count}
    </div>
    {"".join(messages_html)}
</body>
</html>'''

    def to_json(self) -> str:
        """Export conversation to JSON format."""
        import json
        data = {
            "exported": datetime.now().isoformat(),
            "message_count": self.context.message_count,
            "token_count": self.context.token_count,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "tokens": msg.tokens
                }
                for msg in self.context.messages
            ]
        }
        return json.dumps(data, indent=2)

    def to_text(self) -> str:
        """Export conversation to plain text format."""
        lines = [
            f"Conversation Export",
            f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Messages: {self.context.message_count} | Tokens: {self.context.token_count}",
            "",
            "=" * 60,
            ""
        ]

        for msg in self.context.messages:
            role = "You:" if msg.role == "user" else "Assistant:"
            lines.append(role)
            lines.append(msg.content)
            lines.append("")
            lines.append("-" * 40)
            lines.append("")

        return "\n".join(lines)

    def save(self, path: str | Path, format: str = "auto", title: str = "Conversation") -> None:
        """Save conversation to file.

        Args:
            path: Output file path
            format: Export format ('markdown', 'html', 'json', 'text', 'auto')
            title: Title for the export
        """
        path = Path(path)

        # Auto-detect format from extension
        if format == "auto":
            ext = path.suffix.lower()
            format_map = {
                ".md": "markdown",
                ".html": "html",
                ".htm": "html",
                ".json": "json",
                ".txt": "text"
            }
            format = format_map.get(ext, "markdown")

        # Generate content
        if format == "markdown":
            content = self.to_markdown(title)
        elif format == "html":
            content = self.to_html(title)
        elif format == "json":
            content = self.to_json()
        else:
            content = self.to_text()

        path.write_text(content)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))
