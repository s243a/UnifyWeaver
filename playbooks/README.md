# UnifyWeaver Playbooks

This folder contains executable playbooks for LLM agents to follow when performing tasks with UnifyWeaver.

## What Are Playbooks?

Playbooks are step-by-step instruction documents designed for LLM execution. They provide:
- Clear, deterministic steps an AI can follow
- Expected outputs for verification
- Error handling guidance
- Links to supporting examples and documentation

## Available Playbooks

### Data Source Playbooks
| Playbook | Description |
|----------|-------------|
| [csv_data_source_playbook.md](./csv_data_source_playbook.md) | Compile CSV sources to bash |
| [xml_data_source_playbook.md](./xml_data_source_playbook.md) | Python XML parsing pipeline |
| [json_litedb_playbook.md](./json_litedb_playbook.md) | JSON to LiteDB .NET integration |
| [large_xml_streaming_playbook.md](./large_xml_streaming_playbook.md) | Multi-stage XML streaming |

### C# Compilation Playbooks
| Playbook | Description |
|----------|-------------|
| [csharp_codegen_playbook.md](./csharp_codegen_playbook.md) | Non-recursive C# code generation |
| [csharp_query_playbook.md](./csharp_query_playbook.md) | Recursive queries (Fibonacci) |
| [csharp_xml_fragments_playbook.md](./csharp_xml_fragments_playbook.md) | XML fragment streaming |

### Recursion Playbooks
| Playbook | Description |
|----------|-------------|
| [tree_recursion_playbook.md](./tree_recursion_playbook.md) | Tree traversal patterns |
| [mutual_recursion_playbook.md](./mutual_recursion_playbook.md) | Even/odd mutual recursion |

### Execution Playbooks
| Playbook | Description |
|----------|-------------|
| [parallel_execution_playbook.md](./parallel_execution_playbook.md) | Parallel processing |
| [prolog_generation_playbook.md](./prolog_generation_playbook.md) | Multi-dialect Prolog |
| [powershell_inline_dotnet_playbook.md](./powershell_inline_dotnet_playbook.md) | Inline .NET in PowerShell |

## Running Playbooks

### As an LLM Agent

Simply read the playbook and follow each step:
```
Read and execute the playbook at playbooks/csv_data_source_playbook.md
```

### As a Test Orchestrator

To run playbooks across multiple LLM models for testing, see the **[Test Orchestrator Skill](../docs/ai-skills/test-orchestrator.md)**.

This skill teaches you how to:
- Run models headless (Gemini CLI, Claude CLI)
- Use standard test prompts
- Record results in the test matrix
- Compare difficulty ratings across models

## Testing & Quality

Playbooks are tested across multiple LLM tiers to ensure documentation quality:

- **Test Matrix**: [PLAYBOOK_TEST_MATRIX.md](../docs/development/testing/playbooks/PLAYBOOK_TEST_MATRIX.md)
- **Test Prompts**: [test_prompts.md](../docs/development/testing/playbooks/test_prompts.md)
- **Orchestration Guide**: [test-orchestrator.md](../docs/ai-skills/test-orchestrator.md)

### Quality Metric

If simpler models (Tier 1-2) can execute a playbook successfully, the documentation is highly deterministic. If only advanced models (Tier 5+) succeed, the task genuinely requires intelligence.

## Writing Playbooks

For guidance on creating new playbooks:

- [Playbook Philosophy](../docs/development/playbooks/philosophy.md)
- [Playbook Specification](../docs/development/playbooks/specification.md)
- [Best Practices](../docs/development/playbooks/best_practices.md)

## Supporting Files

The `examples_library/` folder contains bash scripts and code snippets referenced by playbooks.
