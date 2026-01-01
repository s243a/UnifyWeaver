# Proposal: Prompt Context Management for Multi-Model Filing Agent

## Problem

Different LLM architectures have different optimal prompt structures:
- **Top lab models** (Claude, GPT-4): Handle discussion flow well, key info at end
- **Chinese models** (Qwen, DeepSeek, GLM, Minimax): Often better with instruction-first
- **Small/local models**: Need compact, structured prompts due to context limits

Currently, the bookmark filing agent uses a fixed prompt structure regardless of provider.

## Proposed Solution

### 1. Prolog-Based Prompt Rules

Define declarative rules for prompt structure selection:

```prolog
% prompt_style(Model, Style)
% Determines which prompt template to use for a given model

prompt_style(claude, discussion_flow).
prompt_style(anthropic, discussion_flow).
prompt_style(openai, discussion_flow).
prompt_style(gemini, discussion_flow).

% Chinese models prefer instruction-first
prompt_style(Model, instruction_first) :-
    model_family(Model, Family),
    chinese_family(Family).

% Small context models need compact prompts
prompt_style(Model, compact) :-
    context_window(Model, Size),
    Size < 4000.

% Model families
model_family(qwen, qwen).
model_family('qwen2.5', qwen).
model_family(deepseek, deepseek).
model_family('deepseek-v3', deepseek).
model_family(glm, zhipu).
model_family('glm-4', zhipu).
model_family(minimax, minimax).
model_family(moonshot, moonshot).

chinese_family(qwen).
chinese_family(deepseek).
chinese_family(zhipu).
chinese_family(minimax).
chinese_family(moonshot).

% Context windows (approximate)
context_window(phi, 2048).
context_window('phi-3', 4096).
context_window('gemma-2b', 8192).
context_window('llama3.1', 131072).
context_window('qwen2.5', 32768).
```

### 2. Prompt Templates

Three base templates that can be selected by the rules:

#### discussion_flow (Claude/GPT style)
```
## Bookmark to File
{bookmark_title}
{bookmark_url}

## Semantic Search Results
{tree_output}

## Existing Folder Contents (if available)
{folder_contents}

## Your Task
Select the best folder from the candidates marked with ★.
Return JSON: {"folder": "...", "reasoning": "..."}
```

#### instruction_first (Chinese model style)
```
## Instructions
You are a bookmark filing assistant. Your task is to select the best
folder from the semantic search candidates.

## Requirements
- Choose the most specific appropriate folder
- Consider the full path hierarchy
- Return valid JSON format

## Candidates
{tree_output}

## Bookmark
Title: {bookmark_title}
URL: {bookmark_url}

## Response Format
{"folder": "...", "reasoning": "..."}
```

#### compact (Small model style)
```
File bookmark "{bookmark_title}" to best folder.

Top candidates:
{top_3_candidates_flat}

Reply: {"folder": "NAME", "reasoning": "WHY"}
```

### 3. Meta-LLM Calibration (Optional Enhancement)

For models not covered by rules, use a cheap LLM to generate optimal template:

#### Calibration Models (cost-effective, good at structured tasks)
- **Gemini 3.0 Flash** - Via `gemini` CLI, model: `gemini-3-flash-preview`
  - Note: Minimal tool call capability requires 3.0 Flash or 2.5 Pro
  - 2.5 Flash lacks sufficient tool call capability
  - 3.0 Flash is the cheap option with tool use
- **Claude Haiku** - Very cheap, good instruction following
- **Qwen2.5-7B** - Free via Ollama, strong benchmarks

#### Calibration Flow
```
1. User adds new model: `--provider ollama --llm-model mixtral`
2. System checks: No cached template for mixtral
3. Calibration prompt sent to Gemini Flash:
   "Given Mixtral's architecture (Mixture of Experts, 32k context),
    generate optimal prompt structure for bookmark filing task.
    Options: discussion_flow, instruction_first, compact, or custom."
4. Response cached to: config/prompt_templates/mixtral.yaml
5. Future queries use cached template
```

#### Calibration Pairs

**Recommended default**: `gemini-3-flash-preview` via Gemini CLI
- Included in Google's ~$20 USD/month plan
- Proven tool call capability
- Works for all target models

| Target Model | Calibrator | Rationale |
|--------------|------------|-----------|
| All models (default) | Gemini CLI (`gemini-3-flash-preview`) | Included in Google ~$20 USD/m plan |
| All models (alt) | Claude CLI (`claude -p --model haiku`) | Included in Claude Pro (~$20 USD/m) |
| Chinese models (future) | GLM-4 / Qwen | Same family - worth testing later |

**Note**: CLI tools with subscriptions are convenient, but API can also work:
- Flash-tier models (Gemini Flash, Haiku) are less expensive than flagship models via API
- Chinese model APIs (DeepSeek, Qwen, GLM) may cost even less if capable
- Flagship models (Opus, GPT-4/5, o1/o3) - consider avoiding for calibration due to cost
  - Note: Sonnet is mid-tier; sometimes Anthropic's top model, sometimes second to Opus

### 4. Session State with Provenance

When saving answers in direct mode, store full context:

```python
@dataclass
class FilingDecision:
    name: str                    # User-assigned variable name
    answer: str                  # Selected folder
    tree_id: str                 # Folder ID

    # Provenance
    query: str                   # Original bookmark title
    timestamp: datetime
    provider: str                # LLM used
    model: str
    prompt_style: str            # Which template was used

    # Search context
    boost_or: Optional[str]
    boost_and: Optional[str]
    filters: List[str]
    alpha: float

    # Results context
    score: float
    rank: int
    alternatives: List[dict]     # Other candidates considered

    # Reproduction
    prompt_hash: str             # Hash of exact prompt sent
```

### 5. Configuration File

`~/.config/unifyweaver/filing.yaml`:

```yaml
defaults:
  mode: conversational  # or 'direct'
  top_k: 10
  alpha: 0.7

providers:
  claude:
    prompt_style: discussion_flow
    model: sonnet

  ollama:
    prompt_style: auto  # Use Prolog rules
    default_model: qwen2.5:14b

  gemini:
    prompt_style: discussion_flow
    model: gemini-2.0-flash
    use_as_calibrator: true

calibration:
  enabled: true
  calibrator: gemini
  cache_dir: ~/.config/unifyweaver/prompt_cache/

session:
  save_decisions: true
  decision_file: ~/.local/share/unifyweaver/filing_history.jsonl
```

## Implementation Phases

### Phase 1: Prolog Source (Initial Target)

Write rules in Prolog first - easy to prototype and validate logic:

```prolog
% src/rules/prompt_context.pl

:- module(prompt_context, [
    prompt_style/2,
    template_order/2,
    calibrator_for/2
]).

% Base rules
prompt_style(claude, discussion_flow).
prompt_style(openai, discussion_flow).
prompt_style(gemini, discussion_flow).

prompt_style(Model, instruction_first) :-
    model_family(Model, Family),
    chinese_family(Family).

prompt_style(Model, compact) :-
    context_window(Model, Size),
    Size < 4000.

% Template section ordering
template_order(discussion_flow, [bookmark, candidates, context, instructions]).
template_order(instruction_first, [instructions, requirements, candidates, bookmark]).
template_order(compact, [combined_instruction, top_candidates]).

% Calibrator pairing
calibrator_for(Model, gemini_cli) :-
    model_family(Model, ollama).
calibrator_for(Model, Calibrator) :-
    model_family(Model, Family),
    chinese_family(Family),
    cheap_chinese(Calibrator).

cheap_chinese(glm_4_flash).
cheap_chinese(qwen_turbo).

% Calibrator details
calibrator_provider(gemini_cli, gemini).
calibrator_model(gemini_cli, 'gemini-3-flash-preview').
```

Test with SWI-Prolog:
```bash
swipl -g "use_module('src/rules/prompt_context.pl'), prompt_style(qwen, S), write(S), nl, halt."
# Output: instruction_first
```

### Phase 2: Transpile to Targets

Once logic is validated, use UnifyWeaver to transpile:

```bash
# To Python (for bookmark_filing_assistant.py)
python3 -m unifyweaver transpile src/rules/prompt_context.pl --target python \
    -o src/unifyweaver/targets/python_runtime/prompt_context.py

# To embedded lookup (for minimal runtime)
python3 -m unifyweaver transpile src/rules/prompt_context.pl --target python-dict \
    -o config/prompt_rules.json
```

This keeps Prolog as source of truth - modify rules in one place, regenerate targets.

### Phase 3: Template System
- Define three base templates as Jinja2 or string templates
- Template selection uses transpiled rules
- Provider-specific overrides in config

### Phase 4: Session State
- `FilingDecision` dataclass with provenance
- Save/load commands in direct mode
- History file for auditing

### Phase 5: Meta-LLM Calibration (Optional)
- Calibration script for new models
- Results feed back as new Prolog facts
- Regenerate targets after calibration

## Open Questions

1. ~~Should Prolog rules be compiled to Python or interpreted at runtime?~~
   **Resolved**: Prolog is source, transpile to targets as needed.
2. How to handle model version differences (qwen2.5-7b vs qwen2.5-72b)?
   - Could use pattern matching: `model_family('qwen2.5-*', qwen)`
   - Or size-based rules: `large_model(M) :- param_count(M, P), P > 30000000000`
3. Should calibration be automatic or user-triggered?
4. Privacy: Should decision history be opt-in?
5. Which UnifyWeaver targets to prioritize for transpilation?
   - Python (immediate need for filing assistant)
   - JSON/dict (embedded config)
   - Others later as needed

## References

- UnifyWeaver Prolog transpiler: `src/unifyweaver/`
- Current filing assistant: `scripts/bookmark_filing_assistant.py`
- Fuzzy boost system: `scripts/fuzzy_boost.py`

## Future Feature: Cross-Account Folder Migration

### Use Case
User has content in account A (e.g., s243a) that they want to move to account B (e.g., s243a_groups) because:
- Account A is no longer updateable
- Consolidating public content into a shared account
- Reorganizing across account boundaries

### Proposed Workflow

```
1. User specifies: source folder, source account, target account
2. System analyzes:
   - Source folder structure and contents
   - Target account structure
   - Privacy check (flag potentially private content)
3. System suggests:
   - Best target location(s) using semantic search
   - Any parent folders that need to be created
   - Conflicts or duplicates in target
4. User confirms
5. System generates:
   - Pearltrees API calls or manual instructions for the move
```

### Example

```bash
python3 scripts/launch_filing_agent.py --move \
    --source "geometry" --source-account s243a \
    --target-account s243a_groups

# Output:
# Source: s243a/science/Math.../Math.../geometry (id10659804)
# Suggested target: s243a_groups/STEM/Math.../Fields of mathematics
# 
# Contents to move:
#   ✓ Fields of geometry (public)
#   ✓ Analytic Geometry (public)
#   ✓ Differential Geometry (public)
#   ...
# 
# No private content detected. Proceed? [y/N]
```

### Implementation Notes
- Reuses existing semantic search for target location
- Privacy detection could use heuristics (folder names, content patterns)
- May require Pearltrees API integration or generate manual steps
