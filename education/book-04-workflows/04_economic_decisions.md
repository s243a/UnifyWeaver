<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 4: Economic Decision Making

This chapter covers how AI agents make strategic decisions based on cost, speed, and quality trade-offs. Economic awareness transforms agents from script executors into intelligent strategists.

## The Economic Model

Every strategy has three dimensions:

```
           Quality
             /\
            /  \
           /    \
          /      \
         /________\
       Cost      Speed
```

**No strategy wins on all dimensions.** The agent must choose based on:
- Current constraints (time, budget, requirements)
- Task characteristics (data size, complexity)
- Deployment context (development, production)

## Trade-off Dimensions

### Cost

Resources consumed:

| Factor | Low Cost | High Cost |
|--------|----------|-----------|
| Compilation | Simple target | Multiple targets |
| Runtime | Single process | Distributed cluster |
| Dependencies | Standard libraries | External services |
| Memory | Streaming | Full materialization |
| Compute | Sequential | Parallel/GPU |

### Speed

Time to completion:

| Factor | Fast | Slow |
|--------|------|------|
| Compilation | Direct output | Optimization passes |
| Runtime | In-memory | Disk I/O |
| Processing | Local | Network round-trips |
| Algorithm | Heuristic | Exhaustive |

### Quality

Result characteristics:

| Factor | Basic | High |
|--------|-------|------|
| Correctness | Approximate | Exact |
| Robustness | Happy path | Full error handling |
| Maintainability | Quick fix | Clean architecture |
| Performance | Works | Optimized |

## Strategy Templates

### Quick Prototype

```yaml
strategy: quick_prototype
cost: low
speed: fast
quality: basic
when_to_use:
  - Development and testing
  - Exploring approaches
  - Small data samples
trade_offs:
  - May not handle edge cases
  - Not optimized for production
  - Limited error handling
```

### Production Ready

```yaml
strategy: production_ready
cost: medium
speed: medium
quality: high
when_to_use:
  - Deployment to production
  - Customer-facing features
  - Reliability requirements
trade_offs:
  - Longer development time
  - More complex implementation
  - Higher testing overhead
```

### High Performance

```yaml
strategy: high_performance
cost: high
speed: very_fast
quality: high
when_to_use:
  - Large data volumes (>1GB)
  - Real-time requirements
  - Compute-intensive tasks
trade_offs:
  - Complex infrastructure
  - Higher operational cost
  - Requires expertise
```

### Enterprise Scale

```yaml
strategy: enterprise_scale
cost: very_high
speed: scales_horizontally
quality: enterprise
when_to_use:
  - Multi-tenant systems
  - Global distribution
  - Regulatory compliance
trade_offs:
  - Significant infrastructure
  - Operational complexity
  - Long development cycles
```

## Decision Heuristics

Heuristics are rules of thumb that guide strategy selection:

### Data Size Heuristics

```prolog
% Choose strategy based on data size
data_size_heuristic(Size, Strategy) :-
    Size < 1000,
    Strategy = quick_prototype.
data_size_heuristic(Size, Strategy) :-
    Size >= 1000, Size < 100000,
    Strategy = production_ready.
data_size_heuristic(Size, Strategy) :-
    Size >= 100000, Size < 10000000,
    Strategy = high_performance.
data_size_heuristic(Size, Strategy) :-
    Size >= 10000000,
    Strategy = enterprise_scale.
```

### Recursion Depth Heuristics

```prolog
% Choose optimization based on recursion depth
recursion_heuristic(Depth, Optimization) :-
    Depth < 10,
    Optimization = none.
recursion_heuristic(Depth, Optimization) :-
    Depth >= 10, Depth < 100,
    Optimization = memoization.
recursion_heuristic(Depth, Optimization) :-
    Depth >= 100, Depth < 1000,
    Optimization = tail_recursion.
recursion_heuristic(Depth, Optimization) :-
    Depth >= 1000,
    Optimization = iterative_conversion.
```

### Time Constraint Heuristics

```prolog
% Choose strategy based on time budget
time_heuristic(urgent, Strategy) :-
    Strategy = quick_prototype.
time_heuristic(normal, Strategy) :-
    Strategy = production_ready.
time_heuristic(flexible, Strategy) :-
    Strategy = optimized.
```

## Cost-Benefit Analysis

### Target Selection by Cost

| Target | Compilation Cost | Runtime Cost | Best For |
|--------|------------------|--------------|----------|
| Bash | Very Low | Medium | Quick scripts, prototyping |
| AWK | Very Low | Low | Text processing |
| Python | Low | Medium | Complex logic, analytics |
| Go | Medium | Low | Performance, concurrency |
| Rust | High | Very Low | Memory-critical, systems |
| C# | Medium | Low | .NET ecosystem |

### Communication Cost Matrix

| Transport | Latency | Throughput | Use When |
|-----------|---------|------------|----------|
| In-process | ~0 | Very High | Same runtime family |
| Pipes | Low | High | Local, streaming |
| TCP/Socket | Medium | High | Same machine |
| HTTP | High | Medium | Different machines |
| Cloud API | Very High | Variable | Distributed |

## Decision Trees

### Compilation Target Decision

```
Is this a quick prototype?
├── Yes → Use Bash/AWK
└── No → Is performance critical?
    ├── Yes → Is memory constrained?
    │   ├── Yes → Use Rust
    │   └── No → Use Go
    └── No → Is .NET integration needed?
        ├── Yes → Use C#
        └── No → Does it need ML/Analytics?
            ├── Yes → Use Python
            └── No → Use Go or Python
```

### Pipeline Stage Decision

```
What is the primary task?
├── Text parsing/extraction
│   └── AWK (fast, streaming)
├── Complex transformations
│   └── Python (rich libraries)
├── Parallel processing
│   └── Go (goroutines)
├── Memory-efficient aggregation
│   └── Rust (zero-copy)
├── .NET ecosystem integration
│   └── C# or PowerShell (in-process)
└── Semantic/ML operations
    └── Python (NumPy, ML frameworks)
```

## Practical Applications

### Example 1: Log Processing

**Scenario**: Process 10GB of log files daily.

**Quick Analysis**:
- Data size: 10GB (large)
- Frequency: Daily (batch acceptable)
- Complexity: Parse, filter, aggregate

**Strategy Selection**:

```prolog
select_log_strategy(DataSize, Frequency, Strategy) :-
    DataSize > 1000000000,  % > 1GB
    Frequency = daily,
    Strategy = high_performance([
        stage(parse, awk),      % Fast text parsing
        stage(transform, go),    % Parallel processing
        stage(aggregate, rust)   % Memory efficient
    ]).
```

**Economic Justification**:
- AWK: 100K lines/sec, minimal memory
- Go: 8x throughput with goroutines
- Rust: Handles aggregation without OOM

### Example 2: Real-Time Analytics

**Scenario**: Sub-second response for dashboard queries.

**Quick Analysis**:
- Latency: <100ms required
- Data: Pre-aggregated, medium size
- Complexity: Joins, filtering

**Strategy Selection**:

```prolog
select_realtime_strategy(Latency, Strategy) :-
    Latency < 100,  % milliseconds
    Strategy = in_memory([
        cache(materialized_views),
        target(go),  % Fast startup, low latency
        transport(in_process)
    ]).
```

### Example 3: Data Science Pipeline

**Scenario**: Exploratory data analysis with visualization.

**Quick Analysis**:
- Interactivity: High (notebooks)
- Libraries: Pandas, NumPy, Matplotlib
- Iteration: Rapid prototyping

**Strategy Selection**:

```prolog
select_datascience_strategy(Strategy) :-
    Strategy = quick_prototype([
        target(python),
        mode(procedural),
        output(jupyter_notebook)
    ]).
```

## Strategy Composition

Complex workflows compose multiple strategies:

```prolog
% Multi-strategy pipeline
composed_pipeline(Input, Output) :-
    % Stage 1: Quick prototype for parsing
    strategy(quick_prototype, parse, Input, Parsed),

    % Stage 2: High performance for heavy compute
    strategy(high_performance, compute, Parsed, Computed),

    % Stage 3: Production ready for output
    strategy(production_ready, format, Computed, Output).
```

## Economic Playbook Section

Every playbook should include economic guidance:

```markdown
## Strategies

### Strategy A: Quick Development
- **Cost**: Low (single Bash target)
- **Speed**: Fast (~5 seconds compilation)
- **Quality**: Basic (no optimization)
- **When to use**: Development, testing, prototyping
- **Trade-offs**: May hit recursion limits, no memoization

### Strategy B: Optimized Production
- **Cost**: Medium (analysis + compilation)
- **Speed**: Slower compilation (~30 seconds)
- **Quality**: High (tail recursion, memoization)
- **When to use**: Production deployment
- **Trade-offs**: Longer build time

### Strategy C: Distributed Scale
- **Cost**: High (multi-target + orchestration)
- **Speed**: Scales horizontally
- **Quality**: Enterprise
- **When to use**: Large data, high availability
- **Trade-offs**: Infrastructure complexity

**Selection Heuristic**:
- Data < 10K rows → Strategy A
- Data 10K-1M rows → Strategy B
- Data > 1M rows → Strategy C
```

## Agent Decision Framework

AI agents should follow this framework:

### 1. Gather Context

```prolog
gather_context(Task, Context) :-
    estimate_data_size(Task, DataSize),
    identify_constraints(Task, Constraints),
    check_requirements(Task, Requirements),
    Context = context{
        data_size: DataSize,
        constraints: Constraints,
        requirements: Requirements
    }.
```

### 2. Enumerate Strategies

```prolog
enumerate_strategies(Context, Strategies) :-
    findall(S, applicable_strategy(Context, S), Strategies).
```

### 3. Score Strategies

```prolog
score_strategy(Context, Strategy, Score) :-
    cost_score(Context, Strategy, CostScore),
    speed_score(Context, Strategy, SpeedScore),
    quality_score(Context, Strategy, QualityScore),
    weighted_average([CostScore, SpeedScore, QualityScore],
                    Context.weights, Score).
```

### 4. Select Best Strategy

```prolog
select_strategy(Context, BestStrategy) :-
    enumerate_strategies(Context, Strategies),
    maplist(score_strategy(Context), Strategies, Scores),
    pairs_keys_values(Pairs, Scores, Strategies),
    keysort(Pairs, Sorted),
    last(Sorted, _-BestStrategy).
```

### 5. Execute with Monitoring

```prolog
execute_with_monitoring(Strategy, Result) :-
    start_timer(Timer),
    execute_strategy(Strategy, Result),
    stop_timer(Timer, Duration),
    log_execution(Strategy, Duration, Result).
```

## Summary

Economic decision-making transforms AI agents into strategic thinkers:

**Key Principles**:
1. **No free lunch**: Every strategy has trade-offs
2. **Context matters**: Best strategy depends on situation
3. **Heuristics guide**: Rules of thumb simplify decisions
4. **Composition works**: Combine strategies for complex workflows

**Decision Framework**:
1. Gather context (data size, constraints, requirements)
2. Enumerate applicable strategies
3. Score based on cost/speed/quality
4. Select best fit
5. Execute with monitoring

The next chapter covers example libraries and pattern reuse.

---

## Navigation

**←** [Previous: Chapter 3: Pipeline Orchestration](03_pipeline_orchestration) | [📖 Book 4: Workflows](./) | [Next: Chapter 5: Example Libraries →](05_example_libraries)
