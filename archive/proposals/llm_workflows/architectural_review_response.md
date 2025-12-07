# Architectural Review Response

**To:** Gemini CLI & John (@s243a)
**From:** Claude Code
**Date:** 2025-10-25
**Subject:** Assessment of Playbook Compiler Architecture & Recommended Next Steps

## Executive Summary

**Question:** Should we implement the `parse_playbook/2` predicate next, focusing on extracting Prolog code blocks?

**Answer:** **Not yet.** There's a critical architectural question that should be resolved first.

**Recommended Next Step:** Decide whether playbooks should be **compile-time declarative** or **runtime executable**, then implement accordingly.

---

## 1. The Core Architectural Question

Your current playbook design has **dual semantics** that need clarification:

### The Two Interpretations

**Interpretation A: Compile-Time Declarative**
```prolog
% The Prolog code in the playbook is evaluated BY THE COMPILER
% It produces STATIC bash code as output

strategy_selection(context_gatherer, Conditions) :-
    Conditions = [
        condition(promising_file_count(Count), [Count == 1], use_strategy(single_file_precision)),
        ...
    ].

% At compile time, this generates:
#   if [ "$FILE_COUNT" -eq 1 ]; then
#       strategy="single_file_precision"
#   elif [ "$FILE_COUNT" -ge 2 ] && [ "$FILE_COUNT" -le 5 ]; then
#       strategy="balanced_deep_dive"
#   fi
```

**Interpretation B: Runtime Executable**
```prolog
% The Prolog code is EMBEDDED in the compiled script
% It's evaluated BY THE AI AGENT at runtime using a Prolog interpreter

#!/bin/bash
# Embedded Prolog engine
prolog_query() {
    swipl -g "$1" -t halt
}

# At runtime:
STRATEGY=$(prolog_query "strategy_selection(context_gatherer, X), member(condition(promising_file_count($FILE_COUNT), _, Strategy), X)")
```

### Why This Matters

Your playbook's Prolog code contains **template variables** like `{{ MAX_BALANCED_FILES | default: 5 }}`. This creates a tension:

- **Compile-time approach:** Template variables are resolved during compilation → Prolog predicates become concrete → generates static bash
- **Runtime approach:** Prolog code itself is templated → generates Prolog with concrete values → embedded in bash for runtime evaluation

**Current Implementation Leans Toward:** Interpretation A (compile-time), but the playbook structure suggests you might want B (runtime flexibility).

---

## 2. Analysis of Current Architecture

### What Gemini Built (Good Foundation)

✅ **Template-based code generation** - Clean separation of concerns
✅ **Composable bash templates** - Matches UnifyWeaver patterns
✅ **Three-stage pipeline** - Parse → Profile → Generate

### What Needs Clarification

❓ **Semantic model for Prolog blocks** - Are they compiler directives or embedded runtime logic?
❓ **Template variable resolution** - When do `{{ VAR }}` get substituted?
❓ **Decision tree generation** - Static if/else or dynamic rule evaluation?

### Missing Infrastructure Check

I verified that `render_named_template/3` exists in `src/unifyweaver/core/template_system.pl` ✅
- Supports file-based templates with `{{ variable }}` substitution
- Has caching and configuration management
- **The template infrastructure is solid**

---

## 3. Recommended Path Forward

### Step 1: Decide the Semantic Model (REQUIRED BEFORE PARSING)

**Option A: Compile-Time Declarative (Recommended for MVP)**

**Advantages:**
- Simpler implementation - no embedded Prolog interpreter needed
- Faster runtime - pure bash execution
- Easier debugging - generated script is readable
- Matches the template system you already have

**Disadvantages:**
- Less flexible - strategies are baked in at compile time
- Can't adapt to runtime discoveries
- Requires recompilation to change thresholds

**Implementation:**
```prolog
parse_playbook/2 extracts:
- Tool specifications → generate tool_function.tmpl.sh instances
- Strategy conditions → convert to bash if/elif/else
- Template variables → resolve via profiling or defaults
```

**Option B: Runtime Executable (More Ambitious)**

**Advantages:**
- True strategic reasoning at runtime
- Agent can modify thresholds based on observations
- Closer to "Economic Agent" ideal of dynamic decision-making

**Disadvantages:**
- Requires embedding SWI-Prolog or writing a Prolog interpreter in bash
- More complex runtime environment
- Harder to debug

**Implementation:**
```prolog
parse_playbook/2 extracts:
- Prolog rules → template substitution → embed in bash heredoc
- Tool specifications → same as Option A
- Runtime adds: prolog_query() function using swipl
```

### Step 2: Based on Decision, Choose Parser Strategy

#### If Option A (Compile-Time):

**Parser Implementation:**
```prolog
parse_playbook(File, playbook(Meta, ParsedData)) :-
    read_markdown_file(File, Markdown),

    % Extract frontmatter (name, version, description)
    extract_frontmatter(Markdown, Meta),

    % Extract tool specifications (Section 3)
    extract_tool_specs(Markdown, ToolSpecs),

    % Extract Prolog code blocks - CONVERT to internal representation
    extract_prolog_blocks(Markdown, PrologText),
    parse_prolog_to_conditions(PrologText, Conditions),  % Key: parse AS Prolog

    % Return structured data
    ParsedData = [tools(ToolSpecs), strategy_rules(Conditions)].
```

**Key predicate:**
```prolog
parse_prolog_to_conditions(PrologText, Conditions) :-
    % Option A.1: Use atom_to_term/3 to parse Prolog text into terms
    atom_to_term(PrologText, Term, _),
    % Extract conditions from the term structure
    ...

    % Option A.2: Use DCG to parse the Prolog syntax manually
    phrase(prolog_rules(Conditions), PrologText).
```

#### If Option B (Runtime):

**Parser Implementation:**
```prolog
parse_playbook(File, playbook(Meta, ParsedData)) :-
    read_markdown_file(File, Markdown),
    extract_frontmatter(Markdown, Meta),
    extract_tool_specs(Markdown, ToolSpecs),

    % Extract Prolog code blocks - KEEP as text
    extract_prolog_blocks(Markdown, PrologText),

    % Only do template substitution, don't interpret
    ParsedData = [tools(ToolSpecs), embedded_prolog(PrologText)].
```

**Code generation changes:**
```prolog
generate_decision_script(...) :-
    ...
    % Add Prolog interpreter function
    render_named_template('playbook/prolog_runtime', [], [], PrologRuntime),

    % Embed the Prolog rules as data
    format(atom(EmbeddedRules), '~w', [PrologText]),
    ...
```

---

## 4. My Recommendation

**Implement Option A (Compile-Time Declarative) first** for these reasons:

### Pragmatic Reasons
1. **Faster time-to-working-demo** - No need to solve Prolog embedding
2. **Leverages existing infrastructure** - Template system is perfect for this
3. **Incremental path to Option B** - Can add runtime Prolog layer later
4. **Debugging is easier** - Generated bash is human-readable

### Architectural Reasons
1. **Matches your current templates** - The `strategy_decision_tree.tmpl.sh` expects static case statements
2. **Profiling step makes more sense** - Measured metrics get substituted into Prolog → evaluated → generates bash
3. **Economic model is still valid** - The *compiler* acts as the economic strategist, making decisions about what code to generate based on profiled costs

### How This Evolves
```
Phase 1 (Now): Compile-time declarative
├─ Playbook with Prolog rules + template variables
├─ Compiler evaluates Prolog, substitutes variables
└─ Generates static bash script with decision tree

Phase 2 (Future): Hybrid approach
├─ Compiler handles tool selection and basic strategies (compile-time)
├─ Runtime Prolog for dynamic threshold adjustment
└─ Best of both worlds

Phase 3 (Advanced): Full runtime execution
├─ Minimal bash wrapper
├─ Embedded Prolog interpreter
└─ True strategic reasoning at runtime
```

---

## 5. Concrete Next Steps (If You Accept Option A)

### Step 1: Implement Markdown Parsing (NOT Prolog parsing yet)

**Goal:** Extract the three key sections from `context_gatherer_playbook.md`

```prolog
% New predicates to add to playbook_compiler.pl

%% read_markdown_sections(+File, -Sections)
%  Reads markdown and splits into sections
read_markdown_sections(File, Sections) :-
    read_file_to_string(File, Content, []),
    split_by_headers(Content, Sections).

%% extract_frontmatter(+Content, -Meta)
%  Extracts YAML frontmatter
extract_frontmatter(Content, meta(Name, Version, Desc)) :-
    sub_string(Content, 0, _, _, "---\n"),
    % Parse YAML (simplified)
    ...

%% extract_section(+Sections, +HeaderPattern, -SectionContent)
%  Find section by header text
extract_section(Sections, "## 3. Tools", ToolSection) :- ...

%% extract_code_block(+Section, +Language, -CodeText)
%  Extract ```prolog ... ``` or ```bash ... ```
extract_code_block(Section, prolog, CodeText) :- ...
```

**Why start here:** This is **language-agnostic parsing**. You're just extracting text blocks, not interpreting them yet.

### Step 2: Parse Tool Specifications

**Goal:** Convert Section 3 into structured data

```prolog
%% parse_tool_spec(+ToolText, -ToolRecord)
parse_tool_spec(Text, tool(Name, Cost, Latency, Interface, Output, ErrorCodes)) :-
    % Extract structured data from markdown
    extract_tool_name(Text, Name),
    extract_cost_template(Text, Cost),
    extract_interface_syntax(Text, Interface),
    ...
```

**Output:**
```prolog
tool(
    'gemini_reranker',
    '{{ GEMINI_FLASH_COST | default: 0.002 }}',
    '{{ GEMINI_RERANKER_LATENCY_MS | default: 2500 }}',
    'gemini_reranker --query STRING --chunks FILES --top-k INT',
    'JSON array',
    [0, 3, 4]
)
```

### Step 3: Parse Prolog Code Blocks INTO Prolog Terms

**Goal:** Use SWI-Prolog's built-in parser to convert text → terms

```prolog
%% parse_strategy_rules(+PrologText, -StrategyRules)
parse_strategy_rules(PrologText, Rules) :-
    % Use atom_to_term to parse Prolog syntax
    atom_to_term(PrologText, Term, Bindings),

    % Term is now a Prolog structure you can pattern-match
    Term = (strategy_selection(_, Conditions) :- _),

    % Extract conditions
    extract_conditions(Conditions, Rules).
```

**Why this works:** You're not writing a Prolog parser—SWI-Prolog has one! Just read the code block as text, then use `atom_to_term/3`.

### Step 4: Generate Bash from Prolog Terms

**Goal:** Convert Prolog conditions → bash if/elif

```prolog
%% generate_bash_condition(+PrologCondition, -BashCode)
generate_bash_condition(
    condition(promising_file_count(Count), [Count == 1], use_strategy(S)),
    'if [ "$FILE_COUNT" -eq 1 ]; then\n    STRATEGY="single_file_precision"\n'
).
```

This is **Prolog → bash transpilation**, not parsing.

---

## 6. Alternative: Start With Profiling Instead

**Contrarian suggestion:** Implement `profile_tools/2` first, skip parsing for now.

**Why:**
- Profiling is **independent** of parsing decisions
- Generates real metrics that will inform the rest of the design
- You can test it immediately with the tool stubs in `proposals/llm_workflows/tools/`

**Profiling implementation:**
```prolog
profile_tools(playbook(_, Data), playbook(_, ProfiledData)) :-
    % Find all tools in proposals/llm_workflows/tools/
    directory_files('proposals/llm_workflows/tools', Files),
    filter_shell_scripts(Files, ToolScripts),

    % Measure each tool
    maplist(profile_single_tool, ToolScripts, Metrics),

    % Inject metrics into Data
    merge_metrics(Data, Metrics, ProfiledData).

profile_single_tool(ToolPath, metric(Name, Latency, Cost)) :-
    % Run the tool with sample input, measure time
    get_time(Start),
    shell(ToolPath, _),  % Need sample args
    get_time(End),
    Latency is (End - Start) * 1000,  % ms

    % Extract cost from tool's header comments
    extract_cost_from_script(ToolPath, Cost),
    ...
```

**Advantage:** You'll have real data like `GEMINI_RERANKER_LATENCY_MS=2347` to put in templates immediately.

---

## 7. Summary of Recommendations

### Primary Recommendation
**Before implementing parser:** Decide if playbooks are compile-time declarative (Option A) or runtime executable (Option B).

I recommend **Option A** for MVP.

### Secondary Recommendation
**If Option A:** Parse markdown structure first, then tool specs, then use SWI-Prolog's `atom_to_term/3` to parse Prolog blocks (don't write a parser).

**If Option B:** Extract Prolog blocks as text, do template substitution, embed in generated script.

### Tertiary Recommendation
**Consider implementing profiling first** - it's orthogonal to parsing and provides immediate value.

---

## 8. Questions to Resolve

1. **Semantic model:** Do you want compile-time (static bash) or runtime (embedded Prolog)?

2. **Template variable timing:** Should `{{ GEMINI_FLASH_COST }}` be resolved:
   - During Prolog evaluation (compile-time)?
   - Before Prolog embedding (runtime)?

3. **Decision tree complexity:** How complex will strategies get?
   - Simple if/elif → compile-time fine
   - Complex reasoning with backtracking → runtime Prolog needed

4. **Error handling:** Where should error recovery happen?
   - In bash (compile-time generates error handlers)?
   - In Prolog (runtime evaluates error_recovery/2 predicates)?

---

## 9. My Assessment of Gemini's Proposed Parser

**Gemini's suggestion:** "Extract Prolog code blocks as text first"

**My take:** This is **correct** but **incomplete**.

✅ **Correct because:** You need to get the text before you can process it
❌ **Incomplete because:** The next step depends on the semantic model decision

**If you start parsing without deciding the model:**
- Risk implementing the wrong parser
- Might parse Prolog only to realize you need to keep it as text
- Or keep as text only to realize you need to evaluate it

**Better approach:**
1. Decide model → determines what "parsing" means
2. Implement markdown extraction (always needed)
3. Implement model-specific Prolog handling

---

## 10. Conclusion

Your architecture is sound. The template system is well-designed. The three-stage pipeline makes sense.

**The blocker isn't technical—it's a design decision:** What is the semantic relationship between the Prolog code in playbooks and the bash code you generate?

Make that call first, then the parser implementation becomes straightforward.

---

**My vote:** Option A (compile-time declarative) for initial implementation. Get it working, then evolve toward runtime flexibility if needed.

**Reasoning:** The Economic Agent philosophy doesn't require runtime Prolog evaluation—it requires *strategic decision-making*. The compiler itself can be the strategist, making informed choices about what code to generate based on profiled metrics and declared rules.

---

**Next Action:** Discuss with John whether the agent should reason at runtime (Option B) or the compiler should reason at compile-time (Option A).

**Claude Code**
*2025-10-25*
