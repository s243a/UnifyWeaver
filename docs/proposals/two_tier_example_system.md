# Proposal: Two-Tier Example System

## Status
- **Status**: Proposed
- **Author**: John William Creighton (@s243a)
- **Co-Author**: Claude Code (Sonnet 4.5)
- **Date**: 2025-11-17
- **Related**: XML Data Source Playbook development

## Summary

Implement a two-tier example system that balances immediate agent productivity (Tier 1: Quick-Start Examples) with long-term learning and generalization capabilities (Tier 2: Composable Building Blocks).

## Motivation

### Current Situation
The XML data source playbook uses complete, self-contained examples that work well for getting agents operational quickly. However, these monolithic examples:
- Are difficult to generalize from
- Repeat patterns across different playbooks
- Don't teach composable concepts
- Make it hard to mix and match patterns

### Lessons Learned
During the XML playbook development with Gemini, we discovered that:
1. **Complete examples work**: Agents can follow step-by-step instructions successfully
2. **Context matters**: Having everything in one place reduces confusion
3. **Clarity is essential**: Explicit instructions prevent misunderstandings
4. **But**: Agents struggle to extract patterns and apply them to new situations

## Proposed Solution

### Two-Tier Architecture

```
playbooks/
├── examples_library/              # Tier 1: Quick-Start Examples
│   ├── xml_examples.md           # Complete working examples
│   ├── parallel_examples.md      # Self-contained scripts
│   └── csv_examples.md           # Ready to run
│
└── building_blocks/               # Tier 2: Composable Patterns
    ├── README.md                  # Index and learning path
    ├── sources/
    │   ├── pattern_define_python_source.md
    │   ├── pattern_define_csv_source.md
    │   └── pattern_define_xml_source.md
    ├── compilation/
    │   ├── pattern_compile_source.md
    │   ├── pattern_write_bash_output.md
    │   └── pattern_inline_compilation.md
    └── execution/
        ├── pattern_run_bash_script.md
        ├── pattern_capture_output.md
        └── pattern_verify_results.md
```

## Tier 1: Quick-Start Examples

### Purpose
Get agents working immediately with minimal learning curve.

### Characteristics
- **Complete**: Everything needed in one file
- **Self-contained**: No external dependencies or assembly required
- **Executable**: Extract and run workflow
- **Documented**: Clear step-by-step instructions
- **Tested**: Verified working examples

### Format
```markdown
## Record: unifyweaver.execution.example_name

Complete description of what this does.

> [!example-record]
> id: unifyweaver.execution.example_name
> name: example_name
> description: Complete working example

```bash
#!/bin/bash
# Complete self-contained script
# ... everything needed ...
```
```

### Target Audience
- Agents new to UnifyWeaver
- Quick prototyping
- Reference implementations
- Testing and validation

## Tier 2: Composable Building Blocks

### Purpose
Enable pattern learning, generalization, and composition.

### Characteristics
- **Focused**: One concept per building block
- **Reusable**: Can be combined in multiple ways
- **Educational**: Explains the pattern and variations
- **Minimal**: Only essential code
- **Annotated**: Comments explain design decisions

### Format
```markdown
# Pattern: Define Python Source

## Concept
How to define a Python source using the source/3 predicate.

## Minimal Example
```prolog
:- use_module(library(unifyweaver/sources)).
:- use_module(library(unifyweaver/sources/python_source)).

:- source(python, my_function, [
    python_inline("
print('Hello from Python')
")
]).
```

## Variations

### With Parameters
[example]

### With File Input
[example]

## Related Patterns
- pattern_compile_source.md
- pattern_write_bash_output.md
```

### Target Audience
- Agents learning UnifyWeaver patterns
- Composing custom solutions
- Understanding design principles
- Advanced usage

## Learning Path

### Phase 1: Quick Success (Tier 1)
```
Agent starts → Uses complete example → Gets working result → Builds confidence
```

### Phase 2: Pattern Recognition (Both Tiers)
```
Agent reviews → Sees Tier 1 reference to patterns → Explores Tier 2 → Understands components
```

### Phase 3: Composition (Tier 2)
```
Agent needs custom solution → Combines building blocks → Creates new pattern → Success
```

### Phase 4: Contribution (Tier 1)
```
Agent creates new Tier 1 example → Documents patterns used → Links to Tier 2 → Completes cycle
```

## Implementation Plan

### Phase 1: Foundation (Current Release)
- [x] Create `docs/proposals/two_tier_example_system.md`
- [x] Document existing Tier 1 examples (xml_examples.md, parallel_examples.md)
- [ ] Add cross-references from Tier 1 to document which patterns they use

### Phase 2: Extract Patterns (Next Release)
- [ ] Create `playbooks/building_blocks/` directory structure
- [ ] Extract patterns from xml_examples.md:
  - `pattern_define_python_source.md`
  - `pattern_compile_source.md`
  - `pattern_write_bash_output.md`
- [ ] Create building_blocks/README.md with index and learning path

### Phase 3: Expand Coverage (Future)
- [ ] Extract patterns from other playbooks
- [ ] Create patterns for common tasks:
  - CSV sources
  - XML sources
  - Compilation workflows
  - Testing patterns
- [ ] Add "Pattern Used" metadata to Tier 1 examples

### Phase 4: Agent Education (Future)
- [ ] Update skills documents to reference both tiers
- [ ] Create "Learning Path" guide
- [ ] Add examples of combining patterns
- [ ] Document pattern composition best practices

## Benefits

### For Quick-Start Examples (Tier 1)
- **Faster onboarding**: Agents get working quickly
- **Reduced errors**: Complete examples are less error-prone
- **Clear expectations**: Known working output
- **Reference implementations**: Canonical examples

### For Building Blocks (Tier 2)
- **Better generalization**: Agents learn transferable patterns
- **Flexibility**: Mix and match for custom solutions
- **Understanding**: Focus on one concept at a time
- **Maintainability**: Changes to patterns propagate

### For the Project
- **Documentation quality**: Clear separation of concerns
- **Agent capability**: Both quick wins and deep learning
- **Pattern library**: Reusable components
- **Evolution**: Easy to add new patterns

## Trade-offs

### Complexity
- **Con**: Two systems to maintain
- **Mitigation**: Cross-reference and validate Tier 1 uses Tier 2 patterns

### Duplication
- **Con**: Same concepts in both tiers
- **Mitigation**: Tier 1 links to Tier 2 for "learn more"

### Discovery
- **Con**: Agents might not find Tier 2
- **Mitigation**: Skills documents and playbooks reference both tiers

## Success Metrics

### Tier 1 Success
- Agents can run examples without modification
- Time to first success < 5 minutes
- Error rate < 5% following instructions

### Tier 2 Success
- Agents create custom solutions using patterns
- Pattern reuse across multiple solutions
- Reduced questions about "how to do X"

### System Success
- Both tiers referenced in agent workflows
- New examples document patterns used
- Pattern library grows organically

## Future Considerations

### Pattern Discovery
- Tool to extract patterns from working code
- Automatic pattern suggestions
- Pattern compatibility matrix

### Pattern Validation
- Automated testing of building blocks
- Composition validation
- Breaking change detection

### Pattern Evolution
- Version tracking for patterns
- Deprecation process
- Migration guides

## References

- XML Data Source Playbook development (feature/xml-data-source-playbook)
- Gemini handoff documents (docs/proposals/llm_workflows/)
- Current examples: playbooks/examples_library/
- Skills documents: skills/

## Conclusion

The two-tier example system provides:
1. **Immediate productivity** through complete examples (Tier 1)
2. **Long-term capability** through composable patterns (Tier 2)
3. **Flexible learning path** from quick-start to mastery
4. **Sustainable growth** of the pattern library

This approach balances the need for quick agent success (proven with Gemini) with the goal of teaching generalizable patterns for advanced usage.
