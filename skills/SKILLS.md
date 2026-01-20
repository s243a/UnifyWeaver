# UnifyWeaver Skills Hierarchy

This document provides a complete overview of all UnifyWeaver skills organized in a three-level hierarchy: **Master → Sub-Master → Individual**.

## Overview

| Master Skill | Description | Child Count |
|--------------|-------------|-------------|
| `skill_server_tools.md` | Backend services, APIs, IPC | 12 |
| `skill_gui_tools.md` | Frontend/GUI generation | 13 |
| `skill_mindmap_tools.md` | Mindmap organization | 6 |
| Standalone | Various specialized skills | ~15 |

---

## Server Tools (Backend)

```
skill_server_tools.md (MASTER)
├── skill_web_frameworks.md (sub-master)
│   ├── skill_flask_api.md
│   ├── skill_fastapi.md
│   └── skill_express_api.md
├── skill_ipc.md (sub-master)
│   ├── skill_pipe_communication.md
│   ├── skill_rpyc.md
│   └── skill_python_bridges.md
└── skill_infrastructure.md (sub-master)
    ├── skill_deployment.md
    ├── skill_authentication.md
    └── skill_networking.md
```

| Skill | Purpose |
|-------|---------|
| `skill_server_tools.md` | Master for backend services |
| `skill_web_frameworks.md` | Flask, FastAPI, Express generation |
| `skill_flask_api.md` | Flask routes and handlers |
| `skill_fastapi.md` | FastAPI with Pydantic models |
| `skill_express_api.md` | Express.js routers |
| `skill_ipc.md` | Inter-process communication patterns |
| `skill_pipe_communication.md` | TSV/JSON pipe protocols |
| `skill_rpyc.md` | Remote Python Call (network RPC) |
| `skill_python_bridges.md` | Cross-runtime Python embedding |
| `skill_infrastructure.md` | Deployment, auth, networking |
| `skill_deployment.md` | Docker, K8s, cloud functions |
| `skill_authentication.md` | Auth backends, JWT, sessions |
| `skill_networking.md` | HTTP/socket servers and clients |

---

## GUI Tools (Frontend)

```
skill_gui_tools.md (MASTER)
├── skill_gui_generation.md (sub-master)
│   ├── skill_typescript_target.md
│   └── skill_component_library.md
├── skill_gui_design.md (sub-master)
│   ├── skill_layout_system.md
│   ├── skill_responsive_design.md
│   └── skill_theming.md
├── skill_gui_runtime.md (sub-master)
│   ├── skill_data_binding.md
│   ├── skill_webassembly.md
│   └── skill_browser_python.md
├── skill_accessibility.md
└── skill_frontend_security.md
```

| Skill | Purpose |
|-------|---------|
| `skill_gui_tools.md` | Master for GUI/frontend |
| `skill_gui_generation.md` | Code generation for frontends |
| `skill_typescript_target.md` | TypeScript compilation target |
| `skill_component_library.md` | UI component patterns |
| `skill_gui_design.md` | Visual design patterns |
| `skill_layout_system.md` | Flexbox, Grid, cross-platform |
| `skill_responsive_design.md` | Breakpoints, media queries |
| `skill_theming.md` | Theme variables, dark mode |
| `skill_gui_runtime.md` | Runtime execution patterns |
| `skill_data_binding.md` | React hooks, WebSocket sync |
| `skill_webassembly.md` | LLVM IR to WASM compilation |
| `skill_browser_python.md` | Pyodide for browser Python |
| `skill_accessibility.md` | ARIA, keyboard navigation |
| `skill_frontend_security.md` | Module whitelisting, rate limiting |

---

## Mindmap Tools

```
skill_mindmap_tools.md (MASTER)
├── skill_mindmap_organization.md (sub-master)
│   ├── skill_mindmap_indexing.md
│   ├── skill_mindmap_linking.md
│   └── skill_mindmap_rename.md
├── skill_mindmap_index.md
├── skill_mindmap_references.md
└── skill_mindmap_cross_links.md
```

| Skill | Purpose |
|-------|---------|
| `skill_mindmap_tools.md` | Master for mindmap management |
| `skill_mindmap_organization.md` | Mindmap structure patterns |
| `skill_mindmap_indexing.md` | Index building and querying |
| `skill_mindmap_linking.md` | Cross-mindmap linking |
| `skill_mindmap_rename.md` | Rename with reference updates |
| `skill_mindmap_index.md` | Index store operations |
| `skill_mindmap_references.md` | Reference management |
| `skill_mindmap_cross_links.md` | Cross-link operations |

---

## Data & Query Skills (Standalone)

| Skill | Purpose |
|-------|---------|
| `skill_sql_target.md` | SQL query generation |
| `skill_stream_aggregation.md` | Streaming aggregation patterns |
| `skill_aggregation_patterns.md` | GROUP BY, window functions |
| `skill_fuzzy_search.md` | Fuzzy text matching |
| `skill_json_sources.md` | JSON data source handling |
| `skill_extract_records.md` | Record extraction patterns |

---

## ML & Embeddings Skills (Standalone)

| Skill | Purpose |
|-------|---------|
| `skill_embedding_models.md` | Embedding model configuration |
| `skill_density_explorer.md` | Density-based exploration |
| `skill_train_model.md` | Model training workflows |
| `skill_semantic_inference.md` | Semantic reasoning |
| `skill_hierarchy_objective.md` | Hierarchical optimization |

---

## Bookmark & Filing Skills (Standalone)

| Skill | Purpose |
|-------|---------|
| `skill_bookmark_filing.md` | Bookmark organization |
| `skill_folder_suggestion.md` | Folder recommendations |
| `skill_mst_folder_grouping.md` | MST-based grouping |

---

## Core & Compiler Skills (Standalone)

| Skill | Purpose |
|-------|---------|
| `skill_unifyweaver_compile.md` | Compilation workflow |
| `skill_transpiler_extension.md` | Custom transpiler extensions |
| `skill_app_generation.md` | Full app scaffolding |
| `skill_find_executable.md` | Executable discovery |
| `skill_unifyweaver_environment.md` | Environment configuration |
| `skill_webapp_security.md` | Web application security |

---

## Complete Alphabetical List

```
skill_accessibility.md
skill_aggregation_patterns.md
skill_app_generation.md
skill_authentication.md
skill_bookmark_filing.md
skill_browser_python.md
skill_component_library.md
skill_data_binding.md
skill_density_explorer.md
skill_deployment.md
skill_embedding_models.md
skill_express_api.md
skill_extract_records.md
skill_fastapi.md
skill_find_executable.md
skill_flask_api.md
skill_folder_suggestion.md
skill_frontend_security.md
skill_fuzzy_search.md
skill_gui_design.md
skill_gui_generation.md
skill_gui_runtime.md
skill_gui_tools.md
skill_hierarchy_objective.md
skill_infrastructure.md
skill_ipc.md
skill_json_sources.md
skill_layout_system.md
skill_mindmap_cross_links.md
skill_mindmap_index.md
skill_mindmap_indexing.md
skill_mindmap_linking.md
skill_mindmap_organization.md
skill_mindmap_references.md
skill_mindmap_rename.md
skill_mindmap_tools.md
skill_mst_folder_grouping.md
skill_networking.md
skill_pipe_communication.md
skill_python_bridges.md
skill_responsive_design.md
skill_rpyc.md
skill_semantic_inference.md
skill_server_tools.md
skill_sql_target.md
skill_stream_aggregation.md
skill_theming.md
skill_train_model.md
skill_transpiler_extension.md
skill_typescript_target.md
skill_unifyweaver_compile.md
skill_unifyweaver_environment.md
skill_webapp_security.md
skill_web_frameworks.md
skill_webassembly.md
```

---

## Skill Relationships

### Parent-Child Relationships

Skills follow a three-level hierarchy:
1. **Master** - Top-level domain (e.g., `skill_server_tools.md`)
2. **Sub-Master** - Category within domain (e.g., `skill_web_frameworks.md`)
3. **Individual** - Specific capability (e.g., `skill_flask_api.md`)

### Cross-References

Skills reference each other via:
- **Parent Skill** - The master/sub-master above
- **Sibling Skills** - Other skills at the same level
- **Related Skills** - Skills in other hierarchies with related functionality

### Code References

Each skill documents the underlying Prolog modules in `src/unifyweaver/`.

---

## Adding New Skills

1. Determine the appropriate hierarchy position
2. Create the skill file following the naming convention: `skill_<name>.md`
3. Include standard sections:
   - When to Use
   - Quick Start
   - Detailed Usage
   - Related (Parent, Sibling, Code)
4. Update this SKILLS.md file
5. Update parent skill's hierarchy section
