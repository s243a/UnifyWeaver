# Skill: Model Registry

Discover, select, load, and manage semantic search models for UnifyWeaver.

## When to Use

- User asks "which model should I use?"
- User wants to load project knowledge models
- User wants to find where models are stored
- User needs to check if models are available
- User wants to regenerate a model
- User asks about model configuration or preferences
- User wants to know how to run semantic search (agent, loop, or inference mode)
- User asks "should I use the agent or direct inference?"

## Quick Start

```bash
# List all registered models (with age)
python3 -m unifyweaver.config.model_registry --list

# Get best knowledge model for project understanding
python3 -m unifyweaver.config.model_registry --knowledge

# Prefer newer models
python3 -m unifyweaver.config.model_registry --knowledge --prefer-newer

# Prefer federated over transformer
python3 -m unifyweaver.config.model_registry --knowledge --prefer-federated

# Show models for a task
python3 -m unifyweaver.config.model_registry --task bookmark_filing

# Check model age
python3 -m unifyweaver.config.model_registry --age orthogonal_transformer_multisource

# Load a model
python3 -m unifyweaver.config.model_registry --load pearltrees_federated_nomic

# List available use modes (agent, loop, inference)
python3 -m unifyweaver.config.model_registry --modes

# Get recommended mode for a task
python3 -m unifyweaver.config.model_registry --mode-for bookmark_filing
```

## Commands

### List Models

```bash
# List all production models (excludes holdout/experimental by default)
python3 -m unifyweaver.config.model_registry --list

# Include holdout/experimental models
python3 -m unifyweaver.config.model_registry --list --all

# Only available models (file exists)
python3 -m unifyweaver.config.model_registry --list --available
```

Output shows `[+]` for available models, `[H]` for HuggingFace (downloadable), `[-]` for missing.
Models are sorted with Nomic-based models first (recommended).

### Filter by Type

```bash
# Embedding models only
python3 -m unifyweaver.config.model_registry --list --type embedding

# Projection models only
python3 -m unifyweaver.config.model_registry --list --type federated

# Transformer models (fast inference)
python3 -m unifyweaver.config.model_registry --list --type orthogonal_codebook
```

### Filter by Embedding Model

```bash
# Show only MiniLM-based models (smaller, C#/ONNX compatible)
python3 -m unifyweaver.config.model_registry --list --embedding minilm

# Show only BGE-based models
python3 -m unifyweaver.config.model_registry --list --embedding bge

# Nomic is default/preferred, so these are shown first
python3 -m unifyweaver.config.model_registry --list --embedding nomic
```

### Filter by Scope

```bash
# Multi-account models (all Pearltrees accounts)
python3 -m unifyweaver.config.model_registry --list --scope multi_account

# Single account models
python3 -m unifyweaver.config.model_registry --list --scope single_account

# Domain-specific models
python3 -m unifyweaver.config.model_registry --list --scope domain
```

### Filter by Tags

```bash
# Show alternative/fallback models
python3 -m unifyweaver.config.model_registry --list --tag alternative

# Account-specific models
python3 -m unifyweaver.config.model_registry --list --tag account_specific

# Exclude specific tags
python3 -m unifyweaver.config.model_registry --list --exclude-tag alternative
```

### Get Models for Task

```bash
python3 -m unifyweaver.config.model_registry --task bookmark_filing
```

### Show Missing Models

```bash
python3 -m unifyweaver.config.model_registry --missing bookmark_filing
```

### Get Model Path

```bash
python3 -m unifyweaver.config.model_registry --path pearltrees_federated_nomic
```

### Show Training Command

```bash
python3 -m unifyweaver.config.model_registry --training pearltrees_federated_nomic
```

## Python API

```python
from unifyweaver.config.model_registry import ModelRegistry

registry = ModelRegistry()

# Get a specific model metadata
model = registry.get_model('pearltrees_federated_nomic')
print(f"Type: {model.type}, Dims: {model.dimensions}")

# Get models for a task
models = registry.get_for_task('bookmark_filing')
projection = models.get('projection')
embedding = models.get('embedding')

# Check availability and age
if registry.check_model_available('pearltrees_federated_nomic'):
    path = registry.get_model_path('pearltrees_federated_nomic')
    age = registry.get_model_age_days('pearltrees_federated_nomic')
    print(f"Model at: {path} ({age:.1f} days old)")

# Load a model for inference
engine = registry.load_model('pearltrees_federated_nomic')

# Load best knowledge model with preferences
engine, name = registry.load_knowledge_model(
    prefer_newer=True,           # Prefer recently trained models
    prefer_federated=False,      # Prefer federated over transformer
    prefer_transformer=True,     # Prefer transformer for speed
)
print(f"Loaded: {name}")

# List all models with age
for m in registry.list_models():
    age = registry.get_model_age_days(m.name)
    print(f"{m.name} ({age:.1f}d): {m.description}")
```

## Loading Knowledge Models

The `load_knowledge_model()` method selects the best model for project understanding:

```python
# Default: first available from candidates list
engine, name = registry.load_knowledge_model()

# Prefer newer models (recently retrained)
engine, name = registry.load_knowledge_model(prefer_newer=True)

# Prefer federated models (may have updated cluster structure)
engine, name = registry.load_knowledge_model(prefer_federated=True)

# Prefer transformer (39x faster inference)
engine, name = registry.load_knowledge_model(prefer_transformer=True)
```

### Knowledge Model Candidates

1. `orthogonal_transformer_multisource` - Combined skills+books, fast (39x)
2. `skills_qa_federated` - Skills documentation only
3. `books_qa_federated_nomic` - Books Q&A only

### Selection Logic

- Multi-source models get bonus (+50 score)
- Type preference adds +100 score
- `prefer_newer` subtracts age in days from score
- First available in sorted order is selected

## Configuration Files

### Lookup Order (first found wins)

1. `.UnifyWeaver-models.yaml` - Project root, user-private
2. `.local/models.yaml` - Already gitignored
3. `config/model_registry.yaml` - User's model definitions (gitignored)
4. `config/model_registry_defaults.yaml` - Public defaults (committed)
5. `config/model_defaults.yaml` - Task defaults (committed)

### User Override Example

Create `.UnifyWeaver-models.yaml` in project root:

```yaml
overrides:
  bookmark_filing:
    projection: pearltrees_federated_s243a_groups

private_paths:
  dropbox: ~/Dropbox/UnifyWeaver/models/
```

## Model Types

| Type | Description | Example |
|------|-------------|---------|
| `embedding` | Foundation text encoders | `nomic-embed-text-v1.5` |
| `federated` | Trained projection models | `pearltrees_federated_nomic` |
| `orthogonal_codebook` | Fast transformer models | `orthogonal_transformer_multisource` |

## Task-to-Model Mappings

| Task | Projection | Embedding |
|------|------------|-----------|
| `bookmark_filing` | `pearltrees_federated_nomic` | `nomic-embed-text-v1.5` |
| `folder_suggestion` | `pearltrees_federated_nomic` | `nomic-embed-text-v1.5` |
| `mindmap_linking` | `pearltrees_federated_nomic` | `nomic-embed-text-v1.5` |
| `skills_qa` | `skills_qa_federated` | `nomic-embed-text-v1.5` |

## Use Modes

Three ways to use semantic search, from interactive to programmatic:

| Mode | Description | Requires LLM | Best For |
|------|-------------|--------------|----------|
| `agent` | Interactive with reasoning | Yes | Complex decisions, ambiguous cases |
| `loop` | Batch with optional LLM | Optional | Batch processing, API integration |
| `inference` | Pure semantic search | No | High throughput, programmatic use |

### CLI Commands

```bash
# List available modes
python3 -m unifyweaver.config.model_registry --modes

# Get recommended mode for a task
python3 -m unifyweaver.config.model_registry --mode-for bookmark_filing

# Show command for a mode (without running)
python3 -m unifyweaver.config.model_registry --run inference

# Run inference mode with a query
python3 -m unifyweaver.config.model_registry --run inference --query "quantum computing"

# Run agent mode interactively
python3 -m unifyweaver.config.model_registry --run agent
```

### Python API

```python
from unifyweaver.config.model_registry import ModelRegistry

registry = ModelRegistry()

# Get available modes
modes = registry.get_use_modes()
for name, config in modes.items():
    print(f"{name}: {config['description']}")

# Get recommended mode for a task
mode = registry.get_mode_for_task('bookmark_filing')  # Returns 'loop'

# Get mode for interactive use
mode = registry.get_mode_for_task('bookmark_filing', variant='interactive')  # Returns 'agent'

# Build command for a mode
cmd = registry.get_mode_command('inference', task='bookmark_filing')

# Run inference mode programmatically
output = registry.run_mode('inference', task='bookmark_filing', query='neural networks')

# Create inference loop for batch processing
engine = registry.create_inference_loop(task='bookmark_filing')
for item in items:
    results = engine.infer(item.text, top_k=10)
    # process results
```

### Task Mode Recommendations

```yaml
# In config/model_defaults.yaml
task_modes:
  bookmark_filing:
    recommended: loop        # Default
    interactive: agent       # When user wants conversation
    batch: loop              # For bulk operations
  skills_qa:
    recommended: agent       # Skills benefit from reasoning
    fast: inference          # Quick lookups
```

## Virtual Environments

The registry manages Python virtual environments for running models with correct dependencies.

### Environment Types

| Type | Description | Example |
|------|-------------|---------|
| `system` | System Python (no venv) | Default fallback |
| `venv` | Python virtual environment | `.venv`, `venv` |
| `conda` | Conda environment | `conda:unifyweaver` |

### CLI Commands

```bash
# List configured environments
python3 -m unifyweaver.config.model_registry --envs

# Check specific environment
python3 -m unifyweaver.config.model_registry --env default

# Get recommended environment for a model
python3 -m unifyweaver.config.model_registry --env-for pearltrees_federated_nomic

# Get activation command
python3 -m unifyweaver.config.model_registry --activate default

# Discover environments (requires permission or --force)
python3 -m unifyweaver.config.model_registry --discover-envs --force
```

### Configuration

```yaml
# In config/model_defaults.yaml
environments:
  default:
    type: venv
    path: .venv
    python_version: "3.11"
    allow_install: true      # OK to pip install
    packages:
      numpy: ">=2.0"
      torch: ">=2.0"

  legacy:
    type: venv
    path: .venv38
    python_version: "3.8"
    allow_install: true
    packages:
      numpy: "<2.0,>=1.24"

  # Conda example
  ml_conda:
    type: conda
    name: unifyweaver        # conda activate unifyweaver
    python_version: "3.11"
    allow_install: true

# Control what the system can auto-discover
inference_permissions:
  discover_environments: false  # Set true to allow auto-discovery
  discover_models: true
  auto_activate: false          # Don't auto-activate without asking

# Model-to-environment requirements
environment_selection:
  preference_order: [default, legacy, system]
  model_requirements:
    pearltrees_federated_nomic:
      requires: "numpy>=2.0"
      preferred_env: default
```

### Python API

```python
from unifyweaver.config.model_registry import ModelRegistry

registry = ModelRegistry()

# List environments
for env in registry.list_available_environments():
    print(f"{env['name']}: {'available' if env['available'] else 'missing'}")

# Check environment
status = registry.check_environment_available('default')
if status['available']:
    print(f"Python: {status['python_path']}")

# Get activation command
cmd = registry.get_activation_command('default')  # "source .venv/bin/activate"

# Get recommended environment for a model
env_name = registry.get_environment_for_model('pearltrees_federated_nomic')

# Check if package installation is allowed
if registry.can_install_packages('default'):
    # OK to pip install
    pass

# Discover environments (if permitted)
discovered = registry.discover_environments(force=True)
for env in discovered:
    print(f"Found: {env['name']} (Python {env['python_version']})")
```

### Why Environment Management?

Models saved with numpy 2.x cannot be loaded with numpy 1.x (and vice versa). The registry:

1. Tracks which environments have which package versions
2. Recommends the right environment for each model
3. Provides activation commands
4. Controls whether auto-discovery is permitted
5. Tracks whether package installation is allowed per environment

### Auto-Environment Inference

Run inference with automatic environment detection - the registry finds an environment with numpy 2.x and uses it:

```bash
# Auto-detect environment and run query
python3 -m unifyweaver.config.model_registry --infer pearltrees_federated_nomic --query "machine learning"

# Interactive mode with auto-environment
python3 -m unifyweaver.config.model_registry --infer pearltrees_federated_nomic --interactive

# See which environment would be used (without running)
python3 -m unifyweaver.config.model_registry --infer pearltrees_federated_nomic
```

Example output:
```
Using environment: hf_env
  Python: /home/user/.hf-env/bin/python

Running inference for 'pearltrees_federated_nomic' with query: machine learning...

1. [0.4641] Machine learning algorithms
   ID: 28831 | Cluster: cluster_161
2. [0.4470] Data mining and machine learning software
   ID: 28820 | Cluster: cluster_161
```

### Python API for Auto-Environment

```python
from unifyweaver.config.model_registry import ModelRegistry

registry = ModelRegistry()

# Find compatible environment (requires numpy 2.x)
env_name = registry.find_compatible_environment('pearltrees_federated_nomic')
print(f"Will use: {env_name}")

# Run inference with auto-environment detection
output = registry.run_inference_script(
    'pearltrees_federated_nomic',
    query='quantum computing',
    top_k=5
)
print(output)

# Run interactively
registry.run_inference_script(
    'pearltrees_federated_nomic',
    interactive=True
)

# Run any command with the right environment
registry.run_with_environment(
    ['scripts/my_script.py', '--arg', 'value'],
    model_name='pearltrees_federated_nomic'
)
```

### Smart Environment Selection

The registry scores environments to help you choose the best one:

```bash
# Show ranked environments with scores
python3 -m unifyweaver.config.model_registry --smart-envs

# With specific package requirements
python3 -m unifyweaver.config.model_registry --smart-envs --packages numpy>=2.0 torch
```

Example output:
```
Smart environment detection (requires: numpy>=2.0)

Rank  Score  Env Name             Python     Status
----------------------------------------------------------------------
1     17     hf_env               3.9.5      Python OK, has: numpy>=2.0, writable
2     11     default              3.11       Python OK, needs: numpy>=2.0, writable
3     -19    system               3.8        Python too old

Recommended for installation: hf_env
```

**Scoring factors:**
- +5 Python version >= 3.9 (required for numpy 2.x)
- -10 Python version too old
- +10 per required package already present
- +5 if writable (allow_install: true)
- +3 if project-local (.venv, venv)
- -20 system Python (avoid modifying)

### Environment Setup

```bash
# Create a new virtual environment
python3 -m unifyweaver.config.model_registry --create-env default --dry-run

# Set up environment (create + install packages)
python3 -m unifyweaver.config.model_registry --setup-env default --dry-run

# Install packages to an environment
python3 -m unifyweaver.config.model_registry --install default --packages numpy>=2.0 torch
```

### Python Version Detection

The registry can extract Python version from environment names:

| Name Pattern | Detected Version |
|--------------|-----------------|
| `venv-3.9`   | 3.9             |
| `venv-3.11`  | 3.11            |
| `default39`  | 3.9             |
| `venv311`    | 3.11            |
| `.venv312`   | 3.12            |
| `py39-env`   | 3.9             |

Configure a naming pattern in `model_defaults.yaml`:

```yaml
environment_selection:
  # Use * as placeholder for Python version
  default_env_pattern: "venv-*"  # Creates venv-3.11 for Python 3.11
```

Python API:
```python
registry = ModelRegistry()

# Get default env name for specific Python version
env_name = registry.get_default_env_name("3.11")  # "venv-3.11" if pattern is "venv-*"

# Extract version from name
version = ModelRegistry._extract_python_version_from_name("venv-3.9")  # "3.9"
```

## Related

**Parent Skill:**
- `skill_ml_tools.md` - ML tools sub-master

**Sibling Skills:**
- `skill_train_model.md` - Train federated models
- `skill_semantic_inference.md` - Run inference
- `skill_embedding_models.md` - Embedding model details

**Documentation:**
- `config/model_registry_defaults.yaml` - Default model definitions
- `config/model_defaults.yaml` - Task mappings
- `docs/design/FEDERATED_MODEL_FORMAT.md` - Model format spec
- `docs/design/ORTHOGONAL_CODEBOOK_DESIGN.md` - Fast inference

**Code:**
- `src/unifyweaver/config/model_registry.py` - Registry implementation
