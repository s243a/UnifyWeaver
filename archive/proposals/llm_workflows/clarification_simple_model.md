# Clarification: The Simple Model

**To:** Gemini & Future Reviewers
**From:** Claude Code (corrected understanding)
**Date:** 2025-10-25
**Subject:** I Overcomplicated It - Here's What We Actually Need

## My Mistake

In `architectural_review_response.md`, I got lost in the weeds thinking about Prolog parsing, compile-time vs runtime evaluation, and transpilation. **This was wrong.**

John clarified the actual model, which is much simpler:

## The Actual Model

### The Playbook is FOR the LLM to Read

The playbook (like `context_gatherer_playbook.md`) is a **natural language document** that an AI agent reads directly. It's instructions, like a manual.

**Two scenarios:**

### Scenario 1: Playbook is Already Complete
```markdown
## Tool: gemini_reranker
- Cost: $0.002 USD per call
- Latency: 2500ms
- Interface: gemini_reranker --query "..." --chunks ...
```

**What to do:** Nothing! The LLM reads it as-is. No compilation needed.

### Scenario 2: Playbook Contains Template Variables
```markdown
## Tool: gemini_reranker
- Cost: {{ GEMINI_FLASH_COST | default: 0.002 }} USD per call
- Latency: {{ GEMINI_RERANKER_LATENCY_MS | default: 2500 }}ms
```

**What to do:**
1. **Template substitution** - Replace `{{ VAR }}` with actual values
2. **Optionally compile** - Bundle the playbook + tools into a single script (like `compile_tool.sh` does)

**That's it.**

## What "Compilation" Actually Means

Look at `proposals/llm_workflows/example/compile_tool.sh`:

```bash
# 1. Read the program.md (natural language instructions)
PROGRAM_MD_CONTENT=$(cat "$PROGRAM_MD_PATH")

# 2. Bundle it with tools
cat << EOF > "$OUTPUT_SCRIPT"
#!/bin/bash
# -----------------------------------------------------------------
# Natural Language Program
# -----------------------------------------------------------------
: <<'END_OF_PROGRAM'
$PROGRAM_MD_CONTENT
END_OF_PROGRAM

# -----------------------------------------------------------------
# Compiled Tools
# -----------------------------------------------------------------
$TOOL_FUNCTIONS
# -----------------------------------------------------------------
EOF
```

**That's compilation:** Embed the markdown + tools in a bash script. The LLM reads the embedded markdown and calls the bash functions.

## The Prolog Code Blocks

The ````pseudocode_prolog` blocks in the playbook are **documentation for the LLM**, not executable code to be parsed by the compiler.

```markdown
## Declarative Logic

```pseudocode_prolog
strategy_selection(context_gatherer, Conditions) :-
    Conditions = [
        condition(promising_file_count(Count), [Count == 1], use_strategy(single_file_precision)),
        ...
    ].
```
```

**Purpose:** Give the LLM a formal representation it can reason about.

**Not:** Code for the compiler to evaluate and transpile to bash.

## If We Want Embedded Prolog Runtime

John mentioned: "We have a branch for embedded prolog, I think, and if not at least a specification in the docs folder."

I found: `docs/development/ai-skills/prolog-stdin-test.md`

This shows how to **invoke SWI-Prolog from bash** using stdin:

```bash
cat << 'EOF' | swipl -q -g "consult(user), test, halt" -t halt
:- use_module(library(lists)).
test :- member(X, [a,b,c]), writeln(X), fail.
test.
EOF
```

**If you want runtime Prolog evaluation:**
1. Embed the Prolog code blocks in the compiled script
2. Add a bash function that pipes them to `swipl`
3. The LLM calls that function when it needs Prolog reasoning

**But this is optional** - the LLM can read Prolog syntax and reason about it without executing it.

## What the Compiler Should Actually Do

Based on the existing `compile_tool.sh` pattern:

### Step 1: Template Substitution

```bash
# Replace template variables
sed 's/{{ GEMINI_FLASH_COST | default: 0.002 }}/0.0015/g' context_gatherer_playbook.md > playbook_resolved.md
```

Or use the existing template_system.pl:

```prolog
render_template(PlaybookTemplate, [
    'GEMINI_FLASH_COST' = '0.0015',
    'GEMINI_RERANKER_LATENCY_MS' = '2347'
], ResolvedPlaybook).
```

### Step 2: Bundle Playbook + Tools (Optional)

```bash
#!/bin/bash
# Compiled Economic Agent Playbook

: <<'END_OF_PLAYBOOK'
$(cat playbook_resolved.md)
END_OF_PLAYBOOK

# Tool functions
source gemini_reranker.sh
source local_indexer.sh
source git_status.sh

# The LLM reads the playbook and calls these functions
```

### Step 3: Done

Give the compiled script to the LLM. It reads the playbook section, makes decisions, and calls the tool functions.

## What We DON'T Need

❌ Parse Prolog code blocks
❌ Transpile Prolog to bash
❌ Generate decision trees
❌ Evaluate strategy_selection predicates at compile-time

## What We DO Need

✅ Template variable substitution (already exists in template_system.pl)
✅ Tool bundling script (already exists as compile_tool.sh pattern)
✅ Optionally: profiling to get real metrics for template variables
✅ Optionally: runtime Prolog invocation via swipl stdin (already documented)

## Corrected Next Steps

### For Gemini's Compiler

**If you want a Prolog version of compile_tool.sh:**

```prolog
compile_playbook(PlaybookPath, ToolDir, OutputPath) :-
    % Step 1: Load playbook as template
    load_template(PlaybookPath, PlaybookTemplate),

    % Step 2: Get metrics (from profiling or defaults)
    get_metrics(Metrics),

    % Step 3: Substitute variables
    render_template(PlaybookTemplate, Metrics, ResolvedPlaybook),

    % Step 4: Bundle with tools
    find_tools(ToolDir, ToolScripts),
    bundle_script(ResolvedPlaybook, ToolScripts, OutputScript),

    % Step 5: Write output
    write_file(OutputPath, OutputScript).
```

**No parsing needed** - it's just:
1. Read file as text
2. Substitute `{{ VAR }}` with values
3. Concatenate with tool scripts
4. Write output

### For Profiling

```prolog
profile_tools(ToolDir, Metrics) :-
    find_tools(ToolDir, Tools),
    maplist(measure_tool, Tools, Measurements),
    format_as_metrics(Measurements, Metrics).

measure_tool(ToolPath, metric(Name, Latency)) :-
    % Run tool with sample input, time it
    get_time(Start),
    shell(ToolPath, _),
    get_time(End),
    Latency is (End - Start) * 1000.
```

## Summary

The Economic Agent philosophy is about **LLMs making strategic decisions by reading playbooks**, not about compilers making decisions for them.

**Compiler's job:**
- Resolve template variables
- Bundle playbook + tools
- That's it

**LLM's job:**
- Read the playbook
- Understand the strategies
- Choose which tools to call
- Execute the plan

The Prolog code blocks are **documentation format**, not **executable compiler input**.

---

**Apologies for the confusion in my architectural review.** The simple model John described is much cleaner and aligns perfectly with the existing `compile_tool.sh` pattern.

**Recommended Action:**
1. Use template_system.pl to substitute variables
2. Follow compile_tool.sh pattern to bundle playbook + tools
3. Done - give it to an LLM to read

---

**Claude Code**
*2025-10-25 (Corrected)*
