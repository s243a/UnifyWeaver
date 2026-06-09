# Distributional Fit Policy

This note specifies the policy layer between exact path-statistic distributions and closed-form approximations. It is a companion to `ROOT_ANCHORED_METRICS_SPECIFICATION.md` and `RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md`.

The problem: root-anchored metrics such as `d_wPow` often start with an exact finite histogram over path statistics. Near the root, or on small graphs, that histogram is cheap and should be kept exactly. Deeper in the graph, especially on enwiki, the support can grow enough that the runtime should switch to a compact representation. That switch must be explicit, diagnosable, and user-overridable.

## 1. Distribution state

The runtime should treat a node's path statistic as a representation choice:

```prolog
distribution_state(exact_histogram(Bins)).

distribution_state(hybrid_truncated(
    ExactPrefix,
    TailFamily,
    TailRange,
    TailMass,
    Parameters,
    ErrorBounds)).

distribution_state(parametric(
    Family,
    SupportRange,
    TotalMass,
    Parameters,
    ErrorBounds)).
```

`Bins` and `ExactPrefix` are finite maps from statistic value to mass. The statistic may be a hop count, weighted length, or a tuple such as `(parent_hops, child_hops)`.

For parent-only paths to a fixed root, the support is finite. Tail fits should therefore be finite-support fits, not infinite asymptotic tails:

```prolog
TailRange = range(KPlusOne, DMax)
```

`DMax` may be the exact longest parent path to root, the active finite horizon, or a conservative structural bound. If only a budget `B` is relevant, `DMax` may be `B` for that computation.

## 2. Default representation policy

The initial policy should be conservative:

```prolog
if support_size <= K:
    keep exact_histogram
else:
    keep exact prefix through K
    fit a truncated tail over K+1 .. DMax
```

`K = 10` is a reasonable first default because it keeps small hand-checkable histograms exact while preventing unbounded per-node vectors. It must be a configuration value, not a constant baked into generated code.

The first tail family should be `truncated_geometric` or equivalently a discrete exponential over finite support. It is simple, has closed-form CDF/survival functions, and matches the intuition that longer parent-only paths often decay after the high-signal prefix. Do not assume a Poisson family by default: parent path distributions are finite, graph-constrained, and driven by branching structure rather than independent arrival counts.

Candidate families for later extension:

| Family | Use when |
|--------|----------|
| `truncated_geometric` | Tail decays approximately exponentially after the exact prefix |
| `truncated_discrete_normal` | Mass is concentrated around a mean with lower variance |
| `beta_binomial` | Bounded support with measurable over/under-dispersion |
| `empirical_sketch` | No simple family fits but quantile/CDF accuracy is enough |
| `mixture(Families)` | Topical and administrative regimes visibly mix |

The family choice should be evidence-driven. Simplewiki is a calibration fixture: compute exact parent-only distributions there, fit candidate tails, and measure error. Enwiki is the stress case where the representation switch is expected to matter.

## 3. Metrics are functionals over the state

The representation policy must not assume the query asks for minimum distance. The distribution state is reusable; the reported metric is a functional over it.

Examples:

```prolog
metric_functional(min_support).
metric_functional(bounded_average(Budget)).
metric_functional(reachability_mass(Budget)).
metric_functional(tail_mass(Budget)).
metric_functional(entropy).
metric_functional(weighted_power_mean(N, Budget)).
```

Approximation error should be assessed against the selected functional. A tail fit that preserves reachability mass may still distort entropy; a fit that preserves bounded average may still move the minimum support point. The selector must therefore carry both:

```prolog
representation_policy(...)
metric_functional(...)
```

## 4. User override surface

The long-term design goal is that users can modify the policy without rewriting target code. The compiler should expose a predicate hook that selects or overrides the representation policy.

Proposed directive:

```prolog
:- distribution_fit_policy(Name/Arity, Options).
```

Example:

```prolog
:- distribution_fit_policy(root_path_distribution/3,
     [ exact_support_limit(10),
       default_tail_family(truncated_geometric),
       max_tail_error(0.01),
       selection_predicate(my_distribution_policy/5) ]).
```

The selection predicate has the shape:

```prolog
my_distribution_policy(+Node,
                       +DistributionSummary,
                       +MetricFunctional,
                       +CostSignals,
                       -RepresentationChoice).
```

`RepresentationChoice` is one of:

```prolog
exact_histogram
hybrid_truncated(Family, PrefixLimit)
parametric(Family)
delegate(DefaultPolicy)
```

The compiler may provide a default predicate:

```prolog
default_distribution_policy(Node, Summary, Functional, Signals, Choice)
```

User predicates can call the default and override only selected cases. This keeps the policy extensible without making target adapters depend on project-specific Prolog code.

## 5. Cost and diagnostics

The representation switch should be driven by both cost and value:

```prolog
switch_to_tail_fit if
    support_size > exact_support_limit
    or histogram_memory > memory_budget
    or propagation_cost > cost_budget
and
    estimated_functional_error <= max_tail_error
```

Diagnostics should be emitted into the recurrence strategy trace:

- exact support size;
- chosen family;
- fitted finite support range;
- estimated error for the selected functional;
- fallback reason if no family passed the threshold;
- whether a user selection predicate overrode the default.

The important invariant is that approximation is not silent. A query that uses a closed-form tail should leave a trace explaining why the representation changed.

## 6. Open validation work

The next implementation-facing work is a parity harness:

1. exact parent-only histogram on tiny fixtures;
2. exact parent-only histogram on simplewiki samples;
3. fitted truncated-tail representation over the same nodes;
4. functional-level error checks for `min_support`, `bounded_average`, `reachability_mass`, `tail_mass`, `entropy`, and `weighted_power_mean`;
5. policy-selection tests showing that a user predicate can override the default without changing target code.

Only after those checks pass should the policy be used for enwiki-scale materialisation.
