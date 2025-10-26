# Handoff to Gemini: Playbook Compiler Implementation

**To:** Gemini CLI
**From:** Claude Code
**Date:** 2025-10-25
**Subject:** Integrated playbook_compiler.pl with UnifyWeaver infrastructure - Ready for next phase

## Summary

I've enhanced your playbook_compiler.pl to properly integrate with UnifyWeaver's existing template_system.pl and follow the established patterns from other source compilers. The core architecture is now in place for multi-stage compilation (parse → profile → generate) with support for both bash and markdown output targets.

The compiler is ready for the next phase: implementing the **Prolog reasoning layer** for economic analysis and completing template variable substitution.

---

## What I Changed

### 1. Fixed Imports and Integration

**Before:**
```prolog
:- use_module(library(shell)).
% Called render_named_template but didn't import it
```

**After:**
```prolog
:- use_module(library(shell)).
:- use_module(library(readutil)).
:- use_module('../core/template_system', [
    render_named_template/3,
    render_named_template/4,
    render_template/3
]).
```

**Why:** Your code called `render_named_template/3` but didn't import it. Now it properly uses the existing template_system.pl infrastructure.

### 2. Added Options-Based Interface

**New predicate:**
```prolog
compile_playbook(+PlaybookIn, +Options, -BashScriptOut)
```

**Options supported:**
- `profile_tools(true/false)` - Whether to measure tool performance
- `output_format(bash/markdown)` - Which target to generate
- `tool_dir(Dir)` - Where to find tool scripts
- `include_analysis(true/false)` - Include cost analysis (future)

**Why:** This follows the UnifyWeaver pattern where source compilers accept options (like csv_source, json_source, etc.). Makes the compiler flexible and configurable.

### 3. Implemented Markdown Parsing

**New predicates:**
- `parse_playbook/3` - Main parsing entry point
- `extract_frontmatter/2` - Parse YAML frontmatter (name, version, description)
- `extract_code_blocks/2` - Extract all ` ```lang ... ``` ` blocks
- `find_template_variables/2` - Find all `{{ VAR }}` or `{{ VAR | default: value }}`

**What it extracts:**
```prolog
playbook(
    "path/to/playbook.md",
    [
        metadata([name="context_gatherer", version="4.0.0", ...]),
        content("full markdown text"),
        code_blocks([block("prolog", "..."), block("bash", "...")]),
        template_vars(["GEMINI_FLASH_COST", "LOCAL_INDEXER_LATENCY_MS", ...])
    ]
)
```

**Why:** The compiler needs structured data to work with. This parsing gives us all the pieces we need for the next stages.

### 4. Enhanced Tool Profiling Structure

**New predicates:**
- `profile_tools/3` - Main profiling entry point (with options)
- `is_shell_script/1` - Filter for .sh files
- `profile_single_tool/3` - Measure one tool's performance

**What it does (currently):**
- Finds tool scripts in `proposals/llm_workflows/tools/`
- Returns placeholder metrics: `metric(ToolName, latency(0), cost(0))`
- Adds metrics to playbook data structure

**What it SHOULD do (next phase):**
```prolog
profile_single_tool(ToolDir, ToolScript, metric(ToolName, latency(Ms), cost(USD))) :-
    % 1. Read tool's header comments to find sample inputs
    % 2. Execute tool with sample inputs
    % 3. Measure execution time with get_time/1
    % 4. Extract cost from header metadata
```

**Why:** This provides the foundation for real metric collection. The structure is there, just needs implementation.

### 5. Added Dual Output Targets

**New predicates:**
- `generate_output/3` - Routes to appropriate generator
- `generate_bash_output/3` - Creates executable bash script
- `generate_markdown_output/3` - Creates resolved markdown playbook

**Bash target:**
- Embeds playbook content in heredoc (using playbook_header.tmpl.sh)
- Will bundle tool functions (TODO)
- Will generate decision tree from strategies (TODO)

**Markdown target:**
- Returns playbook with template variables substituted
- This is the "natural language target" for LLMs
- Currently just returns content as-is (TODO: substitute vars)

**Why:** Different use cases need different outputs. Bash for execution, markdown for LLM reading.

---

## How It Works Now

### Current Pipeline

```
Input: context_gatherer_playbook.md

    ↓ [parse_playbook/3]

Structured playbook:
- metadata: {name, version, description}
- content: full markdown text
- code_blocks: [{lang, code}, ...]
- template_vars: ["VAR1", "VAR2", ...]

    ↓ [profile_tools/3] (if enabled)

Playbook + metrics:
- metrics: [metric(tool, latency, cost), ...]

    ↓ [generate_output/3]

Output (bash or markdown)
```

### What Works
✅ Parsing frontmatter from YAML
✅ Extracting code blocks
✅ Finding template variables
✅ Generating bash wrapper with embedded playbook
✅ Routing to different output formats

### What's Stubbed/TODO
❌ Template variable substitution in markdown output
❌ Actual tool profiling (timing, cost extraction)
❌ Tool bundling in bash output
❌ Decision tree generation from Prolog rules
❌ **Prolog reasoning layer** for cost analysis

---

## What You Should Implement Next

### Priority 1: Template Variable Substitution

**Goal:** Make the markdown output actually substitute `{{ VAR }}` with values.

**Where:** `generate_markdown_output/3` (line ~302)

**Current code:**
```prolog
generate_markdown_output(playbook(_Path, PlaybookData), _Options, ResolvedMarkdown) :-
    memberchk(content(Content), PlaybookData),
    % TODO: Substitute template variables using metrics
    ResolvedMarkdown = Content.  % Just returns as-is
```

**What to implement:**
```prolog
generate_markdown_output(playbook(_Path, PlaybookData), _Options, ResolvedMarkdown) :-
    memberchk(content(Content), PlaybookData),
    memberchk(template_vars(Vars), PlaybookData),

    % Get metrics if available
    (   memberchk(metrics(Metrics), PlaybookData)
    ->  build_substitution_dict(Vars, Metrics, SubstDict)
    ;   build_default_dict(Vars, SubstDict)
    ),

    % Use template_system to substitute
    render_template(Content, SubstDict, ResolvedMarkdown).

% Build dictionary from variables and metrics
build_substitution_dict(Vars, Metrics, Dict) :-
    maplist(map_var_to_metric(Metrics), Vars, Dict).

map_var_to_metric(Metrics, Var, Var=Value) :-
    % Match "GEMINI_RERANKER_LATENCY_MS" to metric(gemini_reranker, latency(2500), _)
    % Extract the 2500 and use it as Value
    ...
```

**Challenge:** Mapping template variable names to metric data structures.

For example:
- `{{ GEMINI_FLASH_COST | default: 0.002 }}` should map to `metric(gemini_reranker, _, cost(0.002))`
- `{{ LOCAL_INDEXER_LATENCY_MS | default: 500 }}` should map to `metric(local_indexer, latency(500), _)`

You might need a naming convention or pattern matching.

### Priority 2: Prolog Reasoning Layer (The Important One!)

**Goal:** Execute Prolog logic from code blocks to pre-compute economic analysis.

**This is what makes the "Economic Agent" economic!**

**Where:** New predicate between profiling and output generation

**Architecture:**
```prolog
compile_playbook(PlaybookIn, Options, Output) :-
    parse_playbook(PlaybookIn, Options, Playbook),
    profile_tools(Playbook, Options, ProfiledPlaybook),

    % NEW: Execute Prolog reasoning layer
    execute_prolog_analysis(ProfiledPlaybook, Options, AnalyzedPlaybook),

    generate_output(AnalyzedPlaybook, Options, Output).
```

**What execute_prolog_analysis should do:**

1. **Extract Prolog code blocks**
```prolog
execute_prolog_analysis(playbook(Path, Data), Options, playbook(Path, EnhancedData)) :-
    memberchk(code_blocks(Blocks), Data),
    findall(Code, member(block("prolog", Code), Blocks), PrologBlocks),
    findall(Code, member(block("pseudocode_prolog", Code), Blocks), PseudoBlocks),
    append(PrologBlocks, PseudoBlocks, AllProlog).
```

2. **Consult the Prolog as a knowledge base**
```prolog
    % Create temporary file with Prolog code
    tmp_file_stream(text, TmpFile, Stream),
    format(Stream, '~s', [AllProlog]),
    close(Stream),

    % Load it
    consult(TmpFile),
```

3. **Execute reasoning queries**
```prolog
    % Get runtime context from options
    option(file_count(N), Options, 50),
    option(budget(B), Options, 0.10),

    % Query the loaded Prolog
    findall(
        strategy(Name, Cost, Quality, Feasible),
        (   strategy(Name, _),
            strategy_cost(Name, N, Cost),
            strategy_quality(Name, Quality),
            (Cost =< B -> Feasible = yes ; Feasible = no)
        ),
        Strategies
    ),
```

4. **Generate analysis results**
```prolog
    % Format as markdown table
    format_cost_analysis_table(Strategies, AnalysisMarkdown),

    % Recommend best strategy
    recommend_strategy(Strategies, B, Recommendation),

    % Add to playbook data
    EnhancedData = [
        analysis(AnalysisMarkdown),
        recommendation(Recommendation)
        | Data
    ].
```

5. **Inject analysis into output**

When generating markdown output, include the analysis:
```prolog
generate_markdown_output(playbook(_Path, Data), _Options, Output) :-
    memberchk(content(Content), Data),

    % If analysis exists, inject it
    (   memberchk(analysis(AnalysisMarkdown), Data)
    ->  % Find {{ COST_ANALYSIS }} placeholder and replace it
        render_template(Content, ['COST_ANALYSIS'=AnalysisMarkdown], Output)
    ;   Output = Content
    ).
```

**Example of what this produces:**

The playbook might have:
```markdown
## Cost Analysis

{{ COST_ANALYSIS }}
```

After Prolog reasoning, this becomes:
```markdown
## Cost Analysis

| Strategy | Cost | Quality | Feasible |
|----------|------|---------|----------|
| Quick Triage | $0.00 | 0.6 | ✓ Yes |
| Balanced Deep Dive | $0.002 | 0.85 | ✓ Yes |

**Recommendation:** Use Balanced Deep Dive for optimal quality/cost ratio.
```

**This is the key value proposition** - the compiler does the economic analysis so the LLM doesn't have to.

### Priority 3: Tool Profiling Implementation

**Goal:** Actually measure tool latency and extract costs.

**Where:** `profile_single_tool/3` (line ~241)

**Implementation:**
```prolog
profile_single_tool(ToolDir, ToolScript, metric(ToolName, latency(Ms), cost(USD))) :-
    format(atom(ToolPath), '~w/~w', [ToolDir, ToolScript]),

    % Extract tool name
    atom_string(ToolScript, ToolStr),
    sub_string(ToolStr, 0, _, 3, ToolNameStr),
    atom_string(ToolName, ToolNameStr),

    % Read tool's header to find cost metadata
    read_file_to_string(ToolPath, ToolContent, []),
    extract_cost_from_header(ToolContent, USD),

    % Measure execution time (would need sample inputs)
    % For now, placeholder:
    Ms = 0,

    format('~`│t     Profiled ~w: ~wms, $~w~n', [ToolName, Ms, USD]).

extract_cost_from_header(Content, Cost) :-
    % Look for: Cost: {{ GEMINI_FLASH_COST | default: 0.002 }}
    % Or:       Cost: $0.002
    sub_string(Content, _, _, _, "Cost:"),
    % Parse the cost value
    % ...
    Cost = 0.002.  % Placeholder
```

---

## Integration Points with UnifyWeaver

### Using template_system.pl

**For variable substitution:**
```prolog
:- use_module('../core/template_system', [render_template/3]).

% Substitute {{ VAR }} in text
render_template("Cost: {{ COST }}", ['COST'='0.002'], Result).
% Result = "Cost: 0.002"
```

**For loading templates:**
```prolog
:- use_module('../core/template_system', [render_named_template/3]).

% Load and render a template file
render_named_template('playbook/playbook_header',
    [playbook_name='my_playbook', playbook_content='...'],
    [],
    RenderedHeader).
```

### Following Source Compiler Patterns

Look at how other compilers work:
- `src/unifyweaver/sources/csv_source.pl`
- `src/unifyweaver/sources/python_source.pl`

They all follow:
1. Define source specification with `:- source(Type, Name, Options).`
2. Compile to bash using `compile_dynamic_source/3`
3. Use template_system for code generation

The playbook compiler follows this same pattern, just targeting natural language instead of bash.

---

## Testing

### Manual Test

```prolog
% Load the compiler
?- ['src/unifyweaver/compilers/playbook_compiler'].

% Compile the context_gatherer playbook
?- compile_playbook(
    'proposals/llm_workflows/context_gatherer_playbook.md',
    [output_format(markdown)],
    Output
).

% Check the output
?- writeln(Output).
```

### What to look for:
- Does it parse the frontmatter correctly?
- Does it find all template variables?
- Does it extract Prolog code blocks?
- Does the output have variables substituted?

### With Options

```prolog
% Enable profiling
?- compile_playbook(
    'proposals/llm_workflows/context_gatherer_playbook.md',
    [profile_tools(true), tool_dir('proposals/llm_workflows/tools')],
    Output
).
```

---

## Architecture Decisions I Made

### 1. Playbooks are Natural Language Documents

I implemented the "simple model" we discussed:
- Playbooks are FOR LLMs to read
- Compilation = variable substitution + optional bundling
- Prolog blocks are documentation/examples, not code to transpile

### 2. Multi-Target Compilation

The compiler supports both:
- **Bash output:** Executable script with embedded playbook (for automation)
- **Markdown output:** Resolved playbook for LLM consumption

This aligns with UnifyWeaver's polyglot compilation vision.

### 3. Optional Prolog Reasoning

The Prolog analysis is optional (not required for basic compilation). You can:
- Compile without analysis → just substitute variables
- Compile with analysis → execute Prolog logic and inject results

This makes the compiler flexible for different use cases.

---

## Suggested Next Steps

**Immediate (finish current feature):**
1. Implement template variable substitution (Priority 1)
2. Test with context_gatherer_playbook.md
3. Fix any parsing issues that come up

**Near-term (core value):**
4. Implement Prolog reasoning layer (Priority 2)
5. Add cost analysis table generation
6. Test with different file counts and budgets

**Future (polish):**
7. Implement actual tool profiling (Priority 3)
8. Add tool bundling to bash output
9. Generate decision tree from Prolog strategies
10. Add more output formats (JSON, YAML for config)

---

## Questions for You

1. **Prolog reasoning interface:** Should runtime context (file_count, budget) be passed as options to compile_playbook/3, or read from environment variables, or both?

2. **Template variable naming:** Should we enforce a naming convention like `TOOLNAME_METRIC_TYPE` (e.g., `GEMINI_RERANKER_LATENCY_MS`), or use a more flexible pattern matching approach?

3. **Analysis injection:** Should the cost analysis be injected via a `{{ COST_ANALYSIS }}` placeholder in the playbook, or automatically added to a specific section (like "## Cost Analysis")?

4. **Caching:** Should we cache profiling results (tool metrics) to avoid re-profiling on every compilation? UnifyWeaver's template_system has caching support we could use.

---

## Files Changed

### Modified
- `src/unifyweaver/compilers/playbook_compiler.pl` (312 lines)
  - Added imports for template_system
  - Implemented parsing predicates
  - Added options-based interface
  - Structured for multi-stage compilation

### Created by You (unchanged by me)
- `templates/playbook/playbook_header.tmpl.sh`
- `templates/playbook/tool_function.tmpl.sh`
- `templates/playbook/strategy_decision_tree.tmpl.sh`
- `proposals/llm_workflows/handoff_for_architectural_review.md`

---

## Summary

The playbook compiler now has a solid foundation integrated with UnifyWeaver's infrastructure. The parsing works, the structure is clean, and it follows established patterns.

**The next phase is the interesting part:** implementing the Prolog reasoning layer so the compiler can actually perform economic analysis and generate those cost comparison tables we've been talking about.

That's where you'll bring the "Economic Agent" philosophy to life - having the compiler pre-compute the tradeoffs so the LLM gets clear, actionable guidance.

Looking forward to seeing what you build on this foundation!

---

**Claude Code**
*2025-10-25*
