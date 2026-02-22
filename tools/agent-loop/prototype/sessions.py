"""Session persistence for saving and loading conversations."""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any

from context import ContextManager, Message, ContextBehavior, ContextFormat


@dataclass
class SessionMetadata:
    """Metadata about a saved session."""
    id: str
    name: str
    created: str
    modified: str
    backend: str
    message_count: int
    token_count: int


class SessionManager:
    """Manages saving and loading conversation sessions."""

    def __init__(self, sessions_dir: str | Path | None = None):
        if sessions_dir is None:
            # Default to ~/.agent-loop/sessions
            sessions_dir = Path.home() / ".agent-loop" / "sessions"
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def save_session(
        self,
        context: ContextManager,
        session_id: str | None = None,
        name: str | None = None,
        backend_name: str = "unknown",
        extra: dict | None = None
    ) -> str:
        """Save a conversation session to disk.

        Returns the session ID.
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate name if not provided
        if name is None:
            name = f"Session {session_id}"

        now = datetime.now().isoformat()

        session_data = {
            "metadata": {
                "id": session_id,
                "name": name,
                "created": now,
                "modified": now,
                "backend": backend_name,
                "message_count": context.message_count,
                "token_count": context.token_count,
            },
            "settings": {
                "max_tokens": context.max_tokens,
                "max_messages": context.max_messages,
                "behavior": context.behavior.value,
                "format": context.format.value,
            },
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "tokens": msg.tokens
                }
                for msg in context.messages
            ],
            "extra": extra or {}
        }

        # Save to file
        session_path = self.sessions_dir / f"{session_id}.json"
        session_path.write_text(json.dumps(session_data, indent=2))

        return session_id

    def load_session(self, session_id: str) -> tuple[ContextManager, dict]:
        """Load a conversation session from disk.

        Returns (context_manager, metadata_dict).
        """
        session_path = self.sessions_dir / f"{session_id}.json"

        if not session_path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        data = json.loads(session_path.read_text())

        # Restore settings
        settings = data.get("settings", {})
        behavior = ContextBehavior(settings.get("behavior", "continue"))
        format_ = ContextFormat(settings.get("format", "plain"))

        context = ContextManager(
            max_tokens=settings.get("max_tokens", 100000),
            max_messages=settings.get("max_messages", 50),
            behavior=behavior,
            format=format_
        )

        # Restore messages
        for msg_data in data.get("messages", []):
            context.add_message(
                role=msg_data["role"],
                content=msg_data["content"],
                tokens=msg_data.get("tokens", 0)
            )

        return context, data.get("metadata", {}), data.get("extra", {})

    def list_sessions(self) -> list[SessionMetadata]:
        """List all saved sessions."""
        sessions = []

        for path in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text())
                meta = data.get("metadata", {})
                sessions.append(SessionMetadata(
                    id=meta.get("id", path.stem),
                    name=meta.get("name", "Unnamed"),
                    created=meta.get("created", ""),
                    modified=meta.get("modified", ""),
                    backend=meta.get("backend", "unknown"),
                    message_count=meta.get("message_count", 0),
                    token_count=meta.get("token_count", 0)
                ))
            except (json.JSONDecodeError, KeyError):
                # Skip invalid session files
                continue

        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a saved session."""
        session_path = self.sessions_dir / f"{session_id}.json"
        if session_path.exists():
            session_path.unlink()
            return True
        return False

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        session_path = self.sessions_dir / f"{session_id}.json"
        return session_path.exists()

    def update_session_name(self, session_id: str, new_name: str) -> bool:
        """Update the name of a saved session."""
        session_path = self.sessions_dir / f"{session_id}.json"
        if not session_path.exists():
            return False

        data = json.loads(session_path.read_text())
        data["metadata"]["name"] = new_name
        data["metadata"]["modified"] = datetime.now().isoformat()
        session_path.write_text(json.dumps(data, indent=2))
        return True