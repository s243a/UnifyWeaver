#!/bin/bash
# Launch Claude Code as a Bookmark Filing Assistant
#
# Usage:
#   ./scripts/launch_bookmark_filing_agent.sh
#   ./scripts/launch_bookmark_filing_agent.sh "My bookmark title"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AGENT_FILE="$PROJECT_DIR/docs/ai-skills/bookmark-filing-agent.md"

# Build initial prompt
INITIAL_PROMPT="You are the Bookmark Filing Assistant. Your role is defined in: docs/ai-skills/bookmark-filing-agent.md

Please read that file first to understand your capabilities.

Key commands available:
- \`python3 scripts/infer_pearltrees_federated.py --model models/pearltrees_federated_single.pkl --query \"TITLE\" --top-k 10 --tree\` - Get candidates
- \`python3 scripts/bookmark_filing_assistant.py --bookmark \"TITLE\" --provider claude\` - Full LLM recommendation
"

# If a bookmark was provided, add it to the prompt
if [ -n "$1" ]; then
    INITIAL_PROMPT="$INITIAL_PROMPT

The user wants to file this bookmark: \"$1\"

Please run the semantic search and provide your recommendation."
fi

# Launch Claude Code with the prompt (interactive mode)
cd "$PROJECT_DIR"

# Use --continue with a new conversation, providing initial context
# The -p flag makes it print-only, we want interactive
exec claude "$INITIAL_PROMPT"
