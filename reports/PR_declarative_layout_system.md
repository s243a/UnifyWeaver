# Proposal: Declarative Layout System for Mind Map Generation

## Problem Statement

The mind map generator has grown to support multiple layout modes (`auto`, `radial`, `tree`), target software considerations, and user preferences. As options proliferate, the Python CLI becomes unwieldy and the logic for selecting appropriate layouts becomes scattered.

**Current pain points:**
- CLI has 30+ arguments, making it hard to use correctly
- Layout selection logic is hardcoded in Python
- No way to express user preferences persistently
- Different mind map software have different native layout capabilities
- Optimal layout depends on item count, hierarchy depth, and content type

## Proposed Solution

A **declarative preference system** using Prolog that:
1. Captures user preferences and software capabilities as facts/rules
2. Infers optimal layout configurations
3. Generates Python invocation or JSON configuration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Preferences                          │
│  (stored in ~/.unifyweaver/preferences.pl or YAML)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Prolog Inference Engine                      │
│  - Software capability facts                                 │
│  - Layout selection rules                                    │
│  - Size/complexity heuristics                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Output Format  │
                    └─────────────────┘
                     /              \
                    ▼                ▼
        ┌──────────────────┐  ┌──────────────────┐
        │  JSON Config     │  │  Python Script   │
        │  (layout spec)   │  │  (direct call)   │
        └──────────────────┘  └──────────────────┘
                    \                /
                     ▼              ▼
        ┌─────────────────────────────────────────┐
        │     generate_simplemind_map.py          │
        │  (accepts --config or CLI args)         │
        └─────────────────────────────────────────┘
```

## Native Layout Reference (.smmx format)

From analysis of .smmx files, the following layout modes are supported per-topic:

```xml
<!-- Native radial (software positions nodes automatically) -->
<layout mode="radial" direction="auto" flow="default"/>

<!-- Free positioning (manual placement preserved) -->
<layout mode="free" direction="manual" flow="default"/>

<!-- Hierarchical layouts -->
<layout mode="top-down" direction="auto" flow="auto"/>
<layout mode="strict-vertical" direction="auto" flow="default"/>
<layout mode="strict-horizontal" direction="auto" flow="default"/>

<!-- Linear layouts (good for lists, 100+ items) -->
<layout mode="list" direction="auto" flow="auto"/>
<layout mode="h-linear" direction="alternating-compact" flow="auto"/>

<!-- Grid layout -->
<layout mode="matrix" direction="bottom" flow="auto"/>
```

**Direction options:** `auto`, `manual`, `alternating-compact`, `bottom`
**Flow options:** `default`, `auto`

Additional style options that affect layout:
```xml
<style borderstyle="sbsCircle" min-width="130.00" min-height="140.00">
  <font bold="True" scale="0.60"/>
  <textcolor r="128" g="255" b="255"/>
</style>
```

**Key insight:** Each topic can have its own layout mode, allowing mixed layouts within a single map (e.g., radial at root, list for large child clusters).

## Prolog Knowledge Base

### Software Capabilities

```prolog
% Target format and native layout support
% format_capability(Format, FeatureType, Feature)
format_capability(smmx, native_layout, radial).
format_capability(smmx, native_layout, 'top-down').
format_capability(smmx, native_layout, 'strict-vertical').
format_capability(smmx, native_layout, 'strict-horizontal').
format_capability(smmx, native_layout, list).
format_capability(smmx, native_layout, matrix).
format_capability(smmx, native_layout, 'h-linear').
format_capability(smmx, native_layout, free).

format_capability(mm, native_layout, none).  % Requires algorithmic layout
format_capability(xmind, native_layout, radial).
format_capability(xmind, native_layout, 'org-chart').

% Format supports per-topic layout (mixed layouts in one map)
format_capability(smmx, per_topic_layout, true).
format_capability(mm, per_topic_layout, false).
```

### User Preferences

```prolog
% User preference: prefer native layout when available
user_preference(default, layout_source, native).

% User preference: fallback algorithmic layout
user_preference(default, fallback_algorithm, radial).

% User preference: size thresholds for layout modes
user_preference(default, radial_max_items, 50).
user_preference(default, linear_min_items, 100).

% User preference: folder organization
user_preference(default, folder_structure, curated).
user_preference(default, parent_links, true).
user_preference(default, titled_files, true).

% User-specific overrides
user_preference(s243a, target_format, smmx).
```

### Layout Selection Rules

```prolog
% Select layout mode based on format, preferences, and data characteristics
select_layout(Format, ItemCount, LayoutMode, LayoutSource) :-
    user_preference(User, layout_source, native),
    format_capability(Format, native_layout, NativeMode),
    appropriate_for_size(NativeMode, ItemCount),
    LayoutMode = NativeMode,
    LayoutSource = auto.

select_layout(Format, ItemCount, LayoutMode, LayoutSource) :-
    user_preference(User, layout_source, native),
    \+ format_capability(Format, native_layout, _),
    user_preference(User, fallback_algorithm, LayoutMode),
    LayoutSource = algorithmic.

% Size appropriateness heuristics
appropriate_for_size(radial, N) :- N =< 50.
appropriate_for_size('strict-vertical', N) :- N > 50, N =< 100.
appropriate_for_size('strict-horizontal', N) :- N > 50, N =< 100.
appropriate_for_size(list, N) :- N > 100.
appropriate_for_size('h-linear', N) :- N > 100.

% Content-type based layout (future enhancement)
appropriate_for_content(matrix, taxonomy).
appropriate_for_content('org-chart', hierarchy).
appropriate_for_content(radial, brainstorm).
```

## Output Formats

### Option 1: JSON Configuration

```json
{
  "layout": {
    "mode": "radial",
    "source": "auto",
    "direction": "auto",
    "flow": "default"
  },
  "output": {
    "format": "smmx",
    "folder_structure": "curated",
    "parent_links": true,
    "titled_files": true
  },
  "style": {
    "tree_style": "ellipse",
    "pearl_style": "half-round"
  }
}
```

Python accepts via: `--config layout_config.json`

### Option 2: Python Script Generation

Prolog generates executable Python:

```python
#!/usr/bin/env python3
# Auto-generated by UnifyWeaver preference system
from scripts.generate_simplemind_map import generate_recursive

generate_recursive(
    cluster_url="https://www.pearltrees.com/t/hacktivism/id2492215",
    data_path="datasets/full_curated.jsonl",
    output_dir="output/hacktivism/",
    layout="auto",
    curated_folders=True,
    parent_links=True,
    use_titled_files=True,
    tree_style="ellipse",
    pearl_style="half-round"
)
```

### Option 3: CLI Arguments String

Prolog outputs shell command:

```bash
python3 scripts/generate_simplemind_map.py \
  --cluster-url "..." \
  --layout auto \
  --curated-folders \
  --parent-links \
  --titled-files
```

## Implementation Phases

### Phase 1: JSON Config Support
- Add `--config` argument to Python CLI
- JSON schema for configuration
- Merge config with CLI args (CLI overrides)

### Phase 2: Prolog Preference Engine
- Define knowledge base schema
- Implement layout selection rules
- Query interface: `select_config(ClusterURL, Software, Config)`

### Phase 3: Integration
- Prolog generates JSON config
- Python consumes config
- Workflow: `swipl -g "generate_config(...)" | python3 generate_simplemind_map.py --config -`

### Phase 4: Adaptive Layout (Future)
- Pre-analyze cluster: item count, depth, content types
- Prolog uses analysis to select optimal layout
- Per-subtree layout rules (different layouts at different depths)

## Example Workflow

```bash
# User sets preferences once
cat > ~/.unifyweaver/preferences.pl << 'EOF'
user_preference(myuser, layout_source, native).
user_preference(myuser, target_format, smmx).
user_preference(myuser, folder_structure, curated).
user_preference(myuser, parent_links, true).
EOF

# Generate with preferences
unifyweaver generate \
  --cluster-url "https://example.com/cluster/id12345" \
  --apply-preferences

# Or override specific settings
unifyweaver generate \
  --cluster-url "..." \
  --apply-preferences \
  --layout radial  # Override to algorithmic
```

## Layout Strategy by Use Case

### For Sharing/Presentation (New Users)
**Recommended:** `--layout auto`
- Delegates to native radial layout engine
- Works well out of the box
- Recipients don't need to know the software to view nicely-arranged maps

### For Aesthetic Control (Advanced Users)
**Options:** `--layout radial`, `--layout tree`, or custom algorithms
- Recreate dense, tightly-packed aesthetics (e.g., original Pearltrees 1 style)
- Control spacing, node density, specific visual goals
- Full control over node positioning

### Hybrid Workflow
Native layout and algorithmic approaches can be combined iteratively:

1. **Native → Free → Optimize**: Open with auto radial, switch to `free` mode (positions preserved), then apply optimizer for refinement
2. **Algorithmic → Native Polish**: Start with our computed positions, let native engine make final adjustments

This works because switching from radial to `free` mode preserves node positions, enabling iterative refinement.

### Layout Mode Summary
| Mode | Behavior | Use Case |
|------|----------|----------|
| `radial-auto` | Native radial, software positions nodes | Sharing, presentation (default) |
| `radial` | Equal angular spacing per parent | Standard radial tree layout |
| `radial-freeform` | Force-directed optimization | Organic, freeform positioning |
| `radial-PT1` | Dense, tightly-packed radial *(future)* | Pearltrees 1 aesthetic |

## Benefits

1. **Separation of concerns**: Layout logic in Prolog, execution in Python
2. **Persistent preferences**: User sets once, applies everywhere
3. **Extensible**: Add new formats, layouts, rules without changing Python
4. **Explainable**: Prolog can explain why it chose a layout
5. **Testable**: Rules can be unit tested independently
6. **Hybrid workflows**: Combine native and algorithmic approaches iteratively

## Open Questions

1. Should preferences be per-user, per-project, or global?
2. How to handle conflicting preferences?
3. Should Prolog run as a service or invoked per-generation?
4. How to version preference schemas?

## Related Work

- The existing `--curated-folders` and `--parent-links` options are proto-preferences
- Configuration file patterns (`.editorconfig`, `pyproject.toml`) for storing project-level settings
- Rule-based expert systems for automated decision making
