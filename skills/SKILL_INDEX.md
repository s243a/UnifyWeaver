# Skill Index

Available skills for AI agents. Load skills as needed based on user request.

## Mindmap Tools

| Skill | File | Use When |
|-------|------|----------|
| **Mindmap Linking** | `skill_mindmap_linking.md` | Link mindmaps to Pearltrees |
| **MST Folder Grouping** | `skill_mst_folder_grouping.md` | Organize by semantic similarity |
| **Cross-Links** | `skill_mindmap_cross_links.md` | Add links between mindmaps |
| **Folder Suggestion** | `skill_folder_suggestion.md` | Find best folder for mindmap |

## Bookmark Tools

| Skill | File | Use When |
|-------|------|----------|
| **Bookmark Filing** | `skill_bookmark_filing.md` | File bookmarks into Pearltrees hierarchy |

## Data Processing

| Skill | File | Use When |
|-------|------|----------|
| **JSON Sources** | `skill_json_sources.md` | Process JSON data |
| **Extract Records** | `skill_extract_records.md` | Extract structured records |

## Environment

| Skill | File | Use When |
|-------|------|----------|
| **Find Executable** | `skill_find_executable.md` | Locate executables |
| **UnifyWeaver Environment** | `skill_unifyweaver_environment.md` | Environment setup |
| **UnifyWeaver Compile** | `skill_unifyweaver_compile.md` | Compile playbooks |

## Quick Reference

### User says... → Load skill

| Trigger | Skill |
|---------|-------|
| "link mindmap to pearltrees" | `skill_mindmap_linking.md` |
| "enrich mindmap" | `skill_mindmap_linking.md` |
| "organize mindmaps" | `skill_mst_folder_grouping.md` |
| "cluster mindmaps" | `skill_mst_folder_grouping.md` |
| "cross-link mindmaps" | `skill_mindmap_cross_links.md` |
| "rename mindmap" | `skill_mindmap_cross_links.md` |
| "where should this go" | `skill_folder_suggestion.md` |
| "file bookmark" | `skill_bookmark_filing.md` |
| "save bookmark" | `skill_bookmark_filing.md` |

## Documentation References

Skills reference these docs for deeper context:

| Doc | Purpose |
|-----|---------|
| `docs/QUICKSTART_MINDMAP_LINKING.md` | Mindmap linking guide |
| `docs/design/FEDERATED_MODEL_FORMAT.md` | Model format spec |
| `scripts/mindmap/README.md` | Full mindmap tools docs |
