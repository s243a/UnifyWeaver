# Proposal: Unified Component Registry

## Summary

This proposal describes a unified registry infrastructure that generalizes the patterns currently used by sources and bindings. The registry provides a common framework for registering, configuring, querying, and validating components across multiple categories.

**Implementation Strategy**: Build the registry first for the new `runtime` category (semantic/ML components like LDA projection). Once proven, refactor existing `source` and `binding` systems to use it.

## Motivation

Currently, UnifyWeaver has parallel but separate infrastructure for:

| Component | Registry | Storage | Query |
|-----------|----------|---------|-------|
| Sources | `source_type_registry/2` | `dynamic_source_def/3` | `is_dynamic_source/1` |
| Bindings | (implicit in module) | `stored_binding/6` | `binding/6` |

This duplication means:
1. Each new component category requires reimplementing registry logic
2. Inconsistent APIs across categories
3. No unified way to discover/enumerate all components
4. Validation logic is duplicated

Additionally, new component types don't fit cleanly into existing categories:
- **Runtime components** (e.g., LDA projection, embedding providers) - not sources or bindings
- **Transforms** (e.g., query expansion, result filtering) - process data but aren't sources
- **Operators** (e.g., matrix operations, aggregations) - mathematical operations

## Proposed Solution

Create a unified `component_registry` module that provides:

1. **Category registration** - define component categories (source, binding, runtime, etc.)
2. **Type registration** - register component types within categories
3. **Instance declaration** - declare configured instances of components
4. **Unified query API** - consistent interface across all categories
5. **Validation framework** - pluggable validation per category/type
6. **Dependency management** - event-based loading of dependencies

## Architecture

### Component Hierarchy

```
Category (source, binding, runtime, transform, operator)
    └── Type (csv, python, lda_projection, etc.)
            └── Instance (my_csv_source, user_projection, etc.)
                    └── Configuration (file paths, parameters, etc.)
```

### Namespacing

Component names are **atoms** that support hierarchical naming:

```prolog
% Simple names (initial implementation)
declare_component(runtime, semantic_projection, lda_projection, [...]).
declare_component(runtime, embedding_provider, onnx_embedding, [...]).

% Hierarchical names supported via atoms (future)
declare_component(runtime, 'user.projections.semantic', lda_projection, [...]).
declare_component(runtime, 'myproject.embeddings.main', onnx_embedding, [...]).
```

**Phase 1**: Single atoms, no package enforcement
**Future**: Package system with file locations, namespace validation

### Registry Structure

```prolog
% Category metadata
:- dynamic category_def/3.
% category_def(Category, Description, Options)

% Type registration (within a category)
:- dynamic type_registry/4.
% type_registry(Category, Type, Module, TypeOptions)

% Instance storage
:- dynamic component_instance/5.
% component_instance(Category, Name, Type, Config, Metadata)

% Dependency graph
:- dynamic component_depends/2.
% component_depends(Name, DependencyList)
```

## API Design

### Category Management

```prolog
%% define_category(+Category, +Description, +Options)
%  Define a new component category.
%
%  Options:
%    - requires_compilation(bool) - whether instances compile to code
%    - singleton(bool) - only one instance per type allowed
%    - validation_hook(Pred) - custom validation predicate
%
define_category(runtime, "Execution-time components", [
    requires_compilation(false),
    singleton(false)
]).

% Future: source and binding categories
% define_category(source, "Data input components", [...]).
% define_category(binding, "Predicate-to-function mappings", [...]).
```

### Type Registration

```prolog
%% register_component_type(+Category, +Type, +Module, +Options)
%  Register a component type within a category.
%
%  The Module must export:
%    - type_info(-Info) - metadata about the type
%    - validate_config(+Config) - configuration validation
%    - init_component/2 - initialization
%    - invoke_component/4 or compile_component/4 depending on category
%
register_component_type(runtime, lda_projection, lda_projection_module, [
    description("LDA-based query-to-answer projection"),
    requires([numpy])
]).

register_component_type(runtime, onnx_embedding, onnx_embedding_module, [
    description("ONNX-based embedding provider"),
    requires([onnxruntime])
]).
```

### Instance Declaration

```prolog
%% declare_component(+Category, +Name, +Type, +Config)
%  Declare a configured instance of a component.
%
%  Config options common to all components:
%    - initialization(eager|lazy|manual) - when to initialize (default: lazy)
%    - depends([Name1, Name2, ...]) - dependencies that must load first
%
declare_component(runtime, embedding_provider, onnx_embedding, [
    model_path('models/all-MiniLM-L6-v2'),
    initialization(eager)  % Load at startup
]).

declare_component(runtime, semantic_projection, lda_projection, [
    model_file('models/W_matrix.npy'),
    lambda_reg(1.0),
    ridge(1e-6),
    embedding_dim(384),
    initialization(lazy),  % Load on first use
    depends([embedding_provider])  % Requires embedding_provider
]).
```

### Dependency Management (Event-Based)

```prolog
%% ensure_component_ready(+Name)
%  Ensure a component and its dependencies are initialized.
%
ensure_component_ready(Name) :-
    component_instance(_, Name, _, Config, _),
    % First, ensure dependencies are ready
    (   member(depends(Deps), Config)
    ->  maplist(ensure_component_ready, Deps)
    ;   true
    ),
    % Then initialize this component if not already
    ensure_initialized(Name).

%% on_component_ready(+Name)
%  Event emitted when a component finishes initialization.
%  Other components can listen for this.
%
on_component_ready(Name) :-
    findall(Dependent,
            (component_depends(Dependent, Deps), member(Name, Deps)),
            Dependents),
    forall(member(D, Dependents), check_ready_to_init(D)).
```

### Initialization Modes

```prolog
%% Initialization modes:
%  - eager: Initialize at system startup
%  - lazy: Initialize on first invoke_component call (default)
%  - manual: Only initialize when explicitly requested

%% init_eager_components/0
%  Called at startup to initialize eager components.
%
init_eager_components :-
    forall(
        (component_instance(_, Name, _, Config, _),
         member(initialization(eager), Config)),
        ensure_component_ready(Name)
    ).

%% invoke_component/4 handles lazy initialization
invoke_component(Category, Name, Input, Output) :-
    ensure_component_ready(Name),  % Lazy init if needed
    component_instance(Category, Name, Type, Config, _),
    type_registry(Category, Type, Module, _),
    Module:invoke_component(Name, Config, Input, Output).
```

### Query API

```prolog
%% component(+Category, ?Name, ?Type, ?Config)
%  Query components. All args except Category can be unbound.
%
component(Category, Name, Type, Config) :-
    component_instance(Category, Name, Type, Config, _).

%% components_of_type(+Category, +Type, -Instances)
%  Get all instances of a specific type.
%
components_of_type(Category, Type, Names) :-
    findall(Name, component_instance(Category, Name, Type, _, _), Names).

%% list_categories(-Categories)
%% list_types(+Category, -Types)
%% list_components(+Category, -Names)
%  Enumeration predicates.
```

### Validation Framework

```prolog
%% validate_component(+Category, +Name, +Config)
%  Validate component configuration.
%
validate_component(Category, Name, Config) :-
    component_instance(Category, Name, Type, _, _),
    type_registry(Category, Type, Module, _),
    % Type-specific validation
    Module:validate_config(Config),
    % Check dependencies exist
    validate_dependencies(Config),
    % Category-level validation
    category_validate(Category, Config).

validate_dependencies(Config) :-
    (   member(depends(Deps), Config)
    ->  forall(member(D, Deps),
               (component_instance(_, D, _, _, _)
                -> true
                ; throw(error(missing_dependency(D)))))
    ;   true
    ).
```

## Implementation Plan

### Phase 1: Core Registry for Runtime Components (Immediate)

Build the registry infrastructure and apply it to the semantic/ML system:

- [ ] Create `src/unifyweaver/core/component_registry.pl`
  - [ ] Category definition (`define_category/3`)
  - [ ] Type registration (`register_component_type/4`)
  - [ ] Instance declaration (`declare_component/4`)
  - [ ] Query API (`component/4`, `list_components/2`, etc.)
  - [ ] Dependency management (event-based)
  - [ ] Initialization modes (eager/lazy/manual)

- [ ] Create runtime category and types:
  - [ ] `define_category(runtime, ...)`
  - [ ] `lda_projection` type
  - [ ] `onnx_embedding` type (wrapper for existing)

- [ ] Create `src/unifyweaver/runtime/lda_projection.pl`
  - [ ] `type_info/1`
  - [ ] `validate_config/1`
  - [ ] `init_component/2`
  - [ ] `invoke_component/4`

- [ ] Create `src/unifyweaver/targets/python_runtime/projection.py`
  - [ ] `LDAProjection` class
  - [ ] Model loading (numpy, JSON)
  - [ ] `project()` and `project_batch()` methods

### Phase 2: Integration with Semantic Search

- [ ] Update `PtSearcher` to use `LDAProjection` component
- [ ] Add projected search mode to Go embedder (loads W, applies projection)
- [ ] Create training script for computing W from Q-A pairs

### Phase 3: Migrate Sources (Future)

Refactor `dynamic_source_compiler.pl` to use the registry:

```prolog
% Wrapper for backward compatibility
register_source_type(Type, Module) :-
    register_component_type(source, Type, Module, []).

dynamic_source_def(Pred/Arity, Type, Config) :-
    component(source, Pred/Arity, Type, Config).

is_dynamic_source(Pred/Arity) :-
    component(source, Pred/Arity, _, _).
```

Changes required:
- [ ] Define `source` category with `requires_compilation(true)`
- [ ] Update source plugins to implement component interface
- [ ] Add wrapper predicates for backward compatibility
- [ ] Update `compile_dynamic_source/3` to use `compile_component/4`

### Phase 4: Migrate Bindings (Future)

Refactor `binding_registry.pl` to use the registry:

```prolog
% Bindings keyed by Target-Pred
declare_binding(Target, Pred, TargetName, Inputs, Outputs, Options) :-
    declare_component(binding, Target-Pred, Target, [
        target_name(TargetName),
        inputs(Inputs),
        outputs(Outputs)
        | Options
    ]).

binding(Target, Pred, TargetName, Inputs, Outputs, Options) :-
    component(binding, Target-Pred, Target, Config),
    member(target_name(TargetName), Config),
    member(inputs(Inputs), Config),
    member(outputs(Outputs), Config),
    % Options are remaining config items
    subtract(Config, [target_name(_), inputs(_), outputs(_)], Options).
```

Changes required:
- [ ] Define `binding` category
- [ ] Map binding structure to component config
- [ ] Add wrapper predicates for backward compatibility
- [ ] Preserve effect annotations

## Implementation Status

### Core Registry

- [x] `src/unifyweaver/core/component_registry.pl` - Core registry infrastructure
- [x] Category definition (`define_category/3`)
- [x] Type registration (`register_component_type/4`)
- [x] Instance declaration (`declare_component/4`)
- [x] Compile component interface (`compile_component/4`)

### Custom Component Types by Target

Custom component types allow injecting raw target language code as reusable components. Each generates a class/struct with an `invoke()` method.

| Target | Module | Config Options | Status |
|--------|--------|----------------|--------|
| Go | `custom_go.pl` | `code(...)`, `imports([...])` | ✅ Implemented |
| Python | `custom_python.pl` | `code(...)`, `imports([...])` | ✅ Implemented |
| Rust | `custom_rust.pl` | `code(...)`, `uses([...])` | ✅ Implemented |
| C# | `custom_csharp.pl` | `code(...)`, `usings([...])` | ✅ Implemented |

### Usage Example

```prolog
% Declare a custom Python component
declare_component(source, my_transform, custom_python, [
    code("return input.upper()"),
    imports(["typing", "re"])
]).

% Declare a custom Rust component
declare_component(source, my_parser, custom_rust, [
    code("input.parse().unwrap()"),
    uses(["std::str::FromStr"])
]).

% Declare a custom C# component
declare_component(source, my_formatter, custom_csharp, [
    code("return input.ToString();"),
    usings(["System.Text"])
]).
```

### Target Integration

Each target has been integrated with the component registry:

| Target | Init Predicate | Helper Predicates | Status |
|--------|----------------|-------------------|--------|
| Go | `init_go_target/0` | `collect_declared_component/2`, `compile_collected_components/1` | ✅ |
| Python | `init_python_target/0` | `collect_declared_component/2`, `compile_collected_components/1` | ✅ |
| Rust | `init_rust_target/0` | `collect_declared_component/2`, `compile_collected_components/1` | ✅ |
| C# | `init_csharp_target/0` | `collect_declared_component/2`, `compile_collected_components/1` | ✅ |

## Runtime Component Example: LDA Projection

### Prolog Type Module

```prolog
% src/unifyweaver/runtime/lda_projection.pl
:- module(lda_projection_module, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    shutdown_component/1
]).

type_info(info(
    name('LDA Semantic Projection'),
    version('1.0.0'),
    description('Projects query embeddings to answer space using learned W matrix'),
    requires([numpy])
)).

validate_config(Config) :-
    % Required: model_file
    (   member(model_file(File), Config)
    ->  true
    ;   throw(error(missing_required_option(model_file)))
    ),
    % Optional with defaults
    (   member(embedding_dim(D), Config) -> integer(D) ; true ),
    (   member(lambda_reg(L), Config) -> number(L) ; true ),
    (   member(ridge(R), Config) -> number(R) ; true ).

init_component(Name, Config) :-
    member(model_file(File), Config),
    (   exists_file(File)
    ->  format('Initialized LDA projection ~w from ~w~n', [Name, File])
    ;   throw(error(model_file_not_found(File)))
    ).

invoke_component(_Name, Config, query(QueryEmbedding), projected(ProjectedEmbedding)) :-
    member(model_file(File), Config),
    invoke_python_projection(File, QueryEmbedding, ProjectedEmbedding).

invoke_component(_Name, Config, query_batch(QueryEmbeddings), projected_batch(ProjectedEmbeddings)) :-
    member(model_file(File), Config),
    invoke_python_projection_batch(File, QueryEmbeddings, ProjectedEmbeddings).

shutdown_component(Name) :-
    format('Shutdown LDA projection ~w~n', [Name]).
```

### Python Runtime Class

```python
# src/unifyweaver/targets/python_runtime/projection.py

import numpy as np
from typing import Optional, Union, List
import json

class LDAProjection:
    """LDA-based semantic projection for RAG queries.

    Projects query embeddings into answer space using a learned
    transformation matrix W derived from Q-A pairs via LDA.

    See: docs/proposals/SEMANTIC_PROJECTION_LDA.md
    """

    def __init__(self, model_file: str, embedding_dim: int = 384):
        """Initialize projection with trained W matrix.

        Args:
            model_file: Path to W matrix (.npy or .json)
            embedding_dim: Expected embedding dimension
        """
        self.embedding_dim = embedding_dim
        self.W: Optional[np.ndarray] = None
        self.load_model(model_file)

    def load_model(self, model_file: str):
        """Load W matrix from file."""
        if model_file.endswith('.npy'):
            self.W = np.load(model_file)
        elif model_file.endswith('.json'):
            with open(model_file) as f:
                self.W = np.array(json.load(f))
        else:
            raise ValueError(f"Unknown model format: {model_file}")

        if self.W.shape != (self.embedding_dim, self.embedding_dim):
            raise ValueError(
                f"W shape mismatch: expected ({self.embedding_dim}, {self.embedding_dim}), "
                f"got {self.W.shape}"
            )

    def project(self, query: np.ndarray) -> np.ndarray:
        """Project single query embedding to answer space.

        Args:
            query: Query embedding vector (d,)

        Returns:
            Projected embedding vector (d,)
        """
        return self.W @ query

    def project_batch(self, queries: np.ndarray) -> np.ndarray:
        """Project batch of query embeddings.

        Args:
            queries: Query embeddings matrix (n, d)

        Returns:
            Projected embeddings matrix (n, d)
        """
        return queries @ self.W.T

    def projected_similarity(self, query: np.ndarray, doc: np.ndarray) -> float:
        """Compute similarity between projected query and document.

        Args:
            query: Query embedding (d,)
            doc: Document embedding (d,)

        Returns:
            Cosine similarity score
        """
        projected = self.project(query)
        return float(np.dot(projected, doc) /
                    (np.linalg.norm(projected) * np.linalg.norm(doc)))
```

### Usage Example

```prolog
% Declare the component
:- declare_component(runtime, semantic_projection, lda_projection, [
    model_file('models/W_matrix.npy'),
    embedding_dim(384),
    initialization(lazy),
    depends([embedding_provider])
]).

% Use in a query predicate
semantic_search(QueryText, TopK, Results) :-
    % Get query embedding
    invoke_component(runtime, embedding_provider, embed(QueryText), embedding(QueryEmb)),
    % Project to answer space
    invoke_component(runtime, semantic_projection, query(QueryEmb), projected(ProjEmb)),
    % Search with projected embedding
    vector_search(ProjEmb, TopK, Results).
```

## Future Features (Not in Initial Implementation)

### Hot-Reloading

Allow updating components without restart:

```prolog
reload_component(Name) :-
    component_instance(Category, Name, Type, Config, _),
    type_registry(Category, Type, Module, _),
    Module:shutdown_component(Name),
    Module:init_component(Name, Config).
```

### Package System

Hierarchical namespacing with file locations:

```prolog
% Package definition
define_package('myproject.semantic', [
    directory('src/myproject/semantic'),
    exports([semantic_projection, query_expander])
]).

% Qualified names
declare_component(runtime, 'myproject.semantic.projection', lda_projection, [...]).
```

### Component Versioning

Track versions for upgrades:

```prolog
declare_component(runtime, semantic_projection, lda_projection, [
    version('1.2.0'),
    compatible_with(['1.1.0', '1.0.0']),
    ...
]).
```

## Benefits

1. **Consistency** - Same API for all component types
2. **Extensibility** - New categories easy to add
3. **Discovery** - Unified enumeration of all components
4. **Validation** - Pluggable validation framework
5. **Dependencies** - Event-based dependency resolution
6. **Flexibility** - Per-component initialization control

## Related Work

- [RAG_MAPPING_SYSTEM.md](RAG_MAPPING_SYSTEM.md) - Generates Q-A pairs for training
- [SEMANTIC_PROJECTION_LDA.md](SEMANTIC_PROJECTION_LDA.md) - Mathematical foundation for LDA projection
- [BINDING_PREDICATE_PROPOSAL.md](BINDING_PREDICATE_PROPOSAL.md) - Current binding system design

## References

- Fisher, R.A. (1936). Linear Discriminant Analysis
- [Linear Discriminant Regularized Regression](https://arxiv.org/abs/2402.14260) - arXiv 2024
