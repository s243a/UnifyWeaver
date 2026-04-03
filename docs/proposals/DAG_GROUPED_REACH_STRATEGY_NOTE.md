# DAG Grouped Reach Strategy Note

## Context

The synthetic dependency reach benchmark exposed a clear DAG-side gap:

- the DFS targets solve transitive reach counting cheaply
- the current C# query path is still much slower on this workload

One natural next idea is to move from:

- repeated seeded closure plus post-aggregation

to:

- a grouped/project-labeled closure strategy

so that project identity is carried through the recursive executor.

## Goal

For a workload like:

- `project -> direct dependency`
- `dependency -> dependency`

we would like to compute:

- reachable dependency set per project

without rerunning closure independently for every project seed.

## Candidate Strategy

Use a grouped closure model where each recursive row carries:

- project label
- current dependency node
- next reachable dependency node

Conceptually:

```text
(project, source_dep, reachable_dep)
```

and extend reachability while preserving the project label.

This is exactly the kind of workload that suggests
`GroupedTransitiveClosureNode` as the eventual runtime abstraction.

## Immediate Obstacle

With the current runtime shape, a direct grouped-closure encoding would
likely require a project-labeled edge relation of the form:

```text
(project, child_dep, parent_dep)
```

which means duplicating dependency edges per project (or per seed
family).

That can explode input volume:

- many projects
- many shared dependency edges
- large repeated grouped edge sets

So the straightforward encoding is probably not the right final design.

## Investigated Alternative

On the current branch, a global closure experiment was tried as a cheaper
replacement for seeded closure.

Result:

- it regressed badly
- seeded plain closure remained better than global closure on the current
  dependency reach benchmark

So the grouped strategy remains interesting, but it should not be
implemented by simply switching to global closure or by naively
duplicating the full edge relation per project.

## Likely Better Direction

The promising next direction is a runtime-level grouped reach operator
that:

1. preserves project/group labels
2. avoids full grouped edge materialization
3. exploits overlap between project dependency cones
4. still keeps the DAG workload in a DAG-specialized execution class

In other words, this likely wants:

- a better grouped closure runtime path

not just:

- a benchmark wrapper rewrite

## Recommendation

Treat grouped/project-labeled reach as the next serious optimization
candidate for the DAG dependency reach benchmark, but as a **runtime
feature** rather than as a simple benchmark-side transformation.
