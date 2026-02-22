"""Search across conversation sessions."""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator


@dataclass
class SearchResult:
    """A single search result."""
    session_id: str
    session_name: str
    message_index: int
    role: str
    content: str
    match_start: int
    match_end: int
    context_before: str
    context_after: str

    def format(self, max_width: int = 80) -> str:
        """Format result for display."""
        # Highlight the match
        before = self.content[:self.match_start]
        match = self.content[self.match_start:self.match_end]
        after = self.content[self.match_end:]

        # Truncate if too long
        if len(before) > 30:
            before = "..." + before[-27:]
        if len(after) > 30:
            after = after[:27] + "..."

        snippet = f"{before}**{match}**{after}"

        return f"[{self.session_id}] {self.role}: {snippet}"


class SessionSearcher:
    """Search through saved conversation sessions."""

    def __init__(self, sessions_dir: str | Path):
        self.sessions_dir = Path(sessions_dir)

    def search(
        self,
        query: str,
        case_sensitive: bool = False,
        regex: bool = False,
        role_filter: str | None = None,
        limit: int = 50
    ) -> list[SearchResult]:
        """Search for a query across all sessions.

        Args:
            query: Search query (string or regex pattern)
            case_sensitive: Whether search is case-sensitive
            regex: Treat query as regex pattern
            role_filter: Filter by role ('user' or 'assistant')
            limit: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        results = []

        # Compile pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        if regex:
            pattern = re.compile(query, flags)
        else:
            pattern = re.compile(re.escape(query), flags)

        # Search each session
        for session_path in self.sessions_dir.glob("*.json"):
            try:
                session_results = self._search_session(
                    session_path, pattern, role_filter
                )
                results.extend(session_results)

                if len(results) >= limit:
                    break
            except (json.JSONDecodeError, KeyError):
                continue

        return results[:limit]

    def _search_session(
        self,
        session_path: Path,
        pattern: re.Pattern,
        role_filter: str | None
    ) -> list[SearchResult]:
        """Search within a single session file."""
        results = []

        data = json.loads(session_path.read_text())
        session_id = data.get("metadata", {}).get("id", session_path.stem)
        session_name = data.get("metadata", {}).get("name", "Unnamed")

        for i, msg in enumerate(data.get("messages", [])):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Apply role filter
            if role_filter and role != role_filter:
                continue

            # Find all matches in this message
            for match in pattern.finditer(content):
                # Extract context
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)

                results.append(SearchResult(
                    session_id=session_id,
                    session_name=session_name,
                    message_index=i,
                    role=role,
                    content=content,
                    match_start=match.start(),
                    match_end=match.end(),
                    context_before=content[start:match.start()],
                    context_after=content[match.end():end]
                ))

        return results

    def search_in_session(
        self,
        session_id: str,
        query: str,
        case_sensitive: bool = False
    ) -> list[SearchResult]:
        """Search within a specific session."""
        session_path = self.sessions_dir / f"{session_id}.json"
        if not session_path.exists():
            return []

        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(query), flags)

        return self._search_session(session_path, pattern, None)

    def list_sessions_containing(
        self,
        query: str,
        case_sensitive: bool = False
    ) -> list[dict]:
        """List sessions that contain a query (without full results)."""
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(query), flags)

        sessions = []

        for session_path in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(session_path.read_text())
                messages = data.get("messages", [])

                # Check if any message matches
                match_count = 0
                for msg in messages:
                    content = msg.get("content", "")
                    match_count += len(pattern.findall(content))

                if match_count > 0:
                    metadata = data.get("metadata", {})
                    sessions.append({
                        "id": metadata.get("id", session_path.stem),
                        "name": metadata.get("name", "Unnamed"),
                        "match_count": match_count,
                        "message_count": len(messages)
                    })
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by match count
        sessions.sort(key=lambda x: x["match_count"], reverse=True)
        return sessions

    def get_session_stats(self) -> dict:
        """Get statistics about all sessions."""
        total_sessions = 0
        total_messages = 0
        total_tokens = 0

        for session_path in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(session_path.read_text())
                total_sessions += 1
                messages = data.get("messages", [])
                total_messages += len(messages)
                total_tokens += sum(m.get("tokens", 0) for m in messages)
            except (json.JSONDecodeError, KeyError):
                continue

        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_tokens": total_tokens
        }