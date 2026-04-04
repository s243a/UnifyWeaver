# Query Engine Materialization Philosophy

## Summary

The parameterized query engine should own materialization decisions.

The ingestion/parser layer should prefer streaming decoded tuples into the
engine. The engine should then decide whether a plan node or operator needs:

- pure streaming evaluation
- replayable input
- compact retained state
- indexed retained state
- full external materialization as a fallback

External pre-materialization remains supported, but it is the non-preferred
route.

## Core Position

The parser should not eagerly decide to retain all facts.

That decision belongs in the query engine because only the engine knows:

- which operator is executing
- whether a second pass is needed
- whether random access is needed
- whether grouped or recursive state is needed
- whether a compact specialized retained form is enough

So the preferred ownership boundary is:

- parser responsibility: stream decoded tuples
- engine responsibility: retain only the state warranted by the plan

## Why This Matters

Recent profiling showed two distinct strengths of the query engine:

- pruning wins on computationally heavy workloads
- intended lazy/streamed execution should reduce unnecessary retention work

The pruning story is already demonstrated.

The streaming/materialization story has only been partially realized because
some benchmark paths were still building eager structures before the engine got
control of the data.

Moving materialization into the engine makes the benchmark behavior align more
closely with the intended architecture.

## Preferred Order Of Strategies

When the engine has a choice, it should prefer these strategies in roughly this
order:

1. streaming with no retained source copy
2. streaming into compact operator-owned retained state
3. replayable or indexed retained state owned by the engine
4. external pre-materialization owned by the caller

This is a heuristic order, not a hard law. A plan may still choose a more
materialized strategy when repeated access or indexing clearly pays for itself.

## External Materialization

External materialization is still valid when:

- an integration point cannot yet stream into the engine
- a caller already has the facts in memory
- a workload needs a compatibility path first
- debugging or inspection is easier with prebuilt relations

But it should be treated as:

- compatible
- supported
- non-preferred

The preferred long-term path is for the engine to decide what state to retain,
not for the parser or benchmark harness to decide prematurely.
