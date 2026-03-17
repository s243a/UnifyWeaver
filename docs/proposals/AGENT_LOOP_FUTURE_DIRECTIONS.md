# Agent Loop — Future Directions

These features have been identified as valuable extensions to the agent loop but are **deferred** pending:

1. Manual testing of Phases 20–24
2. Identification of generalizations in the declarative layer that would make implementation cleaner
3. Broader UnifyWeaver architectural alignment

## Proposed Features

### Conversation Branching

Fork and switch between conversation branches within a session. Useful for exploring multiple approaches to a problem without losing context.

**Key questions:**
- How does this interact with session save/load?
- Could the branching model be declarative (e.g., a tree structure defined as Prolog facts)?
- Is there a general "versioned state" abstraction that other UnifyWeaver components could share?

### Tool Pipelines

Chain tool outputs as inputs to the next tool, allowing multi-step operations without round-tripping through the model.

**Key questions:**
- Could this reuse UnifyWeaver's existing declarative composition patterns?
- Is there overlap with the `compile_component/4` pipeline concept?
- How does this relate to MCP tool chaining?

### Workspace Awareness

Auto-detect project type (language, framework, build system) and suggest relevant tools, prompts, or agent presets.

**Key questions:**
- Detection rules could be declarative facts (`workspace_indicator("Cargo.toml", rust)`), fitting UnifyWeaver's style
- Could this feed into the skills/template system already in place?
- Overlap with the plugin system's auto-discovery?

### Plugin Marketplace

Discover and install community plugins from a registry.

**Key questions:**
- Registry format and hosting
- Trust/verification model
- Could the plugin spec be extended declaratively?

### Agent Delegation

Spawn sub-agents for parallel tasks with result aggregation.

**Key questions:**
- This is architecturally significant — needs careful design
- How does it interact with context limits, cost tracking, and budgets?
- Could delegation be modeled as a declarative workflow graph?

## Guiding Principle

> These features should emerge from generalizations in UnifyWeaver's declarative layer, not be bolted on as imperative special cases. The right time to implement them is after the declarative patterns have been identified and the existing phases have been thoroughly tested.
