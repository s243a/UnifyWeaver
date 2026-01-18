# Agent Instructions

Instructions for AI agents working on this project.

## Skill System

This project has a skill routing system for common tasks.

**At startup, read:** `skills/SKILL_ROUTER.md`

The router provides:
- Decision tree for identifying which skill to load (1.0, 2.1, 2.1.1 numbering)
- LOAD/GOTO instructions for conditional navigation
- Footnotes for edge cases

**Skill index:** `skills/SKILL_INDEX.md`

## When to Refresh

If you forget a command, get an error, or feel uncertain → Re-read the skill, or applicable documentation or code, as part of the process to resolve the issue.

## Available Skill Domains

| Domain | Router Section | Primary Skills |
|--------|----------------|----------------|
| Mindmap | 2.0 | linking, MST grouping, cross-links, folder suggestion |
| Bookmark | 3.0 | bookmark filing |
| Compile | 4.0 | playbook compilation, environment |
| Data | 5.0 | JSON sources, record extraction |
