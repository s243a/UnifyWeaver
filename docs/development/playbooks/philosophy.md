# Playbook Writing Philosophy

## Motivation
UnifyWeaver playbooks should serve both human maintainers and AI agents, maximizing clarity, conciseness, and adaptability.

## Principles
- **Conciseness**: Avoid unnecessary narrative and embedded code; focus on intent and logical structure.
- **Modularity**: Encapsulate implementation details in separate example databases or project documentation and reference them as needed.
- **Literate Programming Style**: Maintain readable, narrative documentation that explains process and workflow, not just technical steps.
- **Agent-centricity**: Design so LLMs require minimal context and can adapt instructions efficiently.
- **Adaptability**: Make playbooks generic and easy to adapt to different tasks/projects by referencing reusable modules and sample databases.

## Strategy
When writing a playbook:
1. State intent and scope in one or two sentences.
2. Structure for skimmability—use sections, lists, and tables where possible.
3. Reference detailed code, examples, and documentation; minimize inline code.
4. Use troubleshooting and verification sections to highlight common errors and expected outputs.
5. Review for brevity and clarity before merging.

## Examples
- Specification: [specification.md](specification.md)
- Best Practices: [best_practices.md](best_practices.md)

## Evolution
Refine philosophy as new workflows and agent needs arise; encourage community feedback and review.
