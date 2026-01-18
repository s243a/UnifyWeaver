# Skill Router

Decision tree for loading skills. Follow from 1.0, branch based on conditions.

## Preamble: Instructions

**Notation:**
- Numbers in brackets like [1] are footnotes. See **Notes** section at end of document.

**LOAD** means:
- If skill is NOT in context → Read the skill file into context
- If skill IS in context → Proceed with skill instructions [1]

**GOTO X.X** means:
- Jump to that numbered section and continue evaluation

**Evaluation order:**
- Start at 1.0
- Evaluate conditions top-to-bottom within each section
- First matching condition wins
- After LOAD, follow the skill's commands

## 1.0 Entry Point

Parse user request and identify domain.

- 1.1 IF mentions "mindmap" OR ".smmx" → GOTO 2.0
- 1.2 IF mentions "bookmark" OR "file" + "pearltrees" → GOTO 3.0
- 1.3 IF mentions "compile" OR "playbook" → GOTO 4.0
- 1.4 IF mentions "json" OR "extract" OR "records" → GOTO 5.0
- 1.5 ELSE → GOTO 6.0 (list available skills)

## 2.0 Mindmap Domain

- 2.1 IF mentions "link" OR "enrich" OR "connect to pearltrees"
  - 2.1.1 LOAD `skill_mindmap_linking.md`
- 2.2 IF mentions "organize" OR "cluster" OR "group" OR "mst"
  - 2.2.1 LOAD `skill_mst_folder_grouping.md`
- 2.3 IF mentions "cross-link" OR "rename" OR "index" OR "cloudmapref"
  - 2.3.1 LOAD `skill_mindmap_cross_links.md`
- 2.4 IF mentions "suggest" OR "where should" OR "best folder"
  - 2.4.1 LOAD `skill_folder_suggestion.md`
- 2.5 ELSE (general mindmap task)
  - 2.5.1 LOAD `skill_mindmap_linking.md` (default)

## 3.0 Bookmark Domain

- 3.1 IF mentions "file" OR "save" OR "organize" OR "where to put"
  - 3.1.1 LOAD `skill_bookmark_filing.md`
- 3.2 IF mentions "candidates" OR "suggestions"
  - 3.2.1 LOAD `skill_bookmark_filing.md`
- 3.3 ELSE
  - 3.3.1 LOAD `skill_bookmark_filing.md`

## 4.0 Compile Domain

- 4.1 IF mentions "compile" OR "playbook" OR "prolog"
  - 4.1.1 LOAD `skill_unifyweaver_compile.md`
- 4.2 IF mentions "environment" OR "setup"
  - 4.2.1 LOAD `skill_unifyweaver_environment.md`
- 4.3 IF mentions "find" + "executable"
  - 4.3.1 LOAD `skill_find_executable.md`

## 5.0 Data Processing Domain

- 5.1 IF mentions "json" + "source"
  - 5.1.1 LOAD `skill_json_sources.md`
- 5.2 IF mentions "extract" OR "records"
  - 5.2.1 LOAD `skill_extract_records.md`

## 6.0 Fallback

- 6.1 No matching domain identified
- 6.2 Read `SKILL_INDEX.md` for available skills
- 6.3 Ask user to clarify intent

## Notes

[1] If you forget the command, get an error, or feel uncertain → Re-read the skill to refresh, or applicable documentation or code, as part of the process to resolve the issue.

## General Notes

- Multiple conditions can match; load first matching skill
- After loading skill, follow its commands
- Skills reference docs for deeper context (e.g., `--help`, README files)
- If skill doesn't solve problem, return to router and try next branch
