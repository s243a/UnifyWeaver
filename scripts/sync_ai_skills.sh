#!/bin/bash
# sync_ai_skills.sh - Sync AI skills between docs and .claude directories
#
# Usage:
#   ./scripts/sync_ai_skills.sh           # Sync from docs/ to .claude/
#   ./scripts/sync_ai_skills.sh --reverse # Sync from .claude/ to docs/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DOCS_SKILLS_DIR="$PROJECT_ROOT/docs/development/ai-skills"
CLAUDE_SKILLS_DIR="$PROJECT_ROOT/.claude/skills"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure directories exist
mkdir -p "$CLAUDE_SKILLS_DIR"
mkdir -p "$DOCS_SKILLS_DIR"

if [[ "$1" == "--reverse" ]]; then
    # Sync from .claude/skills/ to docs/development/ai-skills/
    echo "Syncing skills: .claude/skills/ → docs/development/ai-skills/"

    for skill_file in "$CLAUDE_SKILLS_DIR"/*.md; do
        if [[ -f "$skill_file" ]]; then
            skill_name=$(basename "$skill_file")
            echo -e "${GREEN}✓${NC} Copying $skill_name to docs"
            cp "$skill_file" "$DOCS_SKILLS_DIR/$skill_name"
        fi
    done
else
    # Sync from docs/development/ai-skills/ to .claude/skills/ (default)
    echo "Syncing skills: docs/development/ai-skills/ → .claude/skills/"

    # Copy all skill markdown files (except README.md)
    for skill_file in "$DOCS_SKILLS_DIR"/*.md; do
        if [[ -f "$skill_file" ]]; then
            skill_name=$(basename "$skill_file")

            # Skip README.md
            if [[ "$skill_name" == "README.md" ]]; then
                echo -e "${YELLOW}i${NC} Skipping README.md"
                continue
            fi

            echo -e "${GREEN}✓${NC} Copying $skill_name to .claude/skills/"
            cp "$skill_file" "$CLAUDE_SKILLS_DIR/$skill_name"
        fi
    done
fi

echo ""
echo "Skill sync complete!"
echo "  Source of truth: docs/development/ai-skills/"
echo "  Active skills:   .claude/skills/"
echo ""
echo "To sync in reverse (from .claude/ to docs/):"
echo "  ./scripts/sync_ai_skills.sh --reverse"
