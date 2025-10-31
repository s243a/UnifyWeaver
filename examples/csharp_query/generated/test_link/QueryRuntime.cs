// SPDX-License-Identifier: MIT OR Apache-2.0
// Generated runtime scaffolding for UnifyWeaver C# query execution
// Provides minimal infrastructure for executing declarative plans
// emitted by the forthcoming csharp_query target.

using System;
using System.Collections.Generic;
using System.Linq;

namespace UnifyWeaver.QueryRuntime
{
    /// <summary>
    /// Predicate identifier consisting of functor name and arity.
    /// </summary>
    public readonly record struct PredicateId(string Name, int Arity)
    {
        public override string ToString() => $"{Name}/{Arity}";
    }

    /// <summary>
    /// Interface that supplies materialised tuples for base relations.
    /// </summary>
    public interface IRelationProvider
    {
        IEnumerable<object[]> GetFacts(PredicateId predicate);
    }

    /// <summary>
    /// Base class for all query plan nodes.
    /// </summary>
    public abstract record PlanNode;

    /// <summary>
    /// Scans a base relation provided by <see cref="IRelationProvider"/>.
    /// </summary>
    public sealed record RelationScanNode(PredicateId Relation) : PlanNode;

    /// <summary>
    /// Applies a tuple-level filter.
    /// </summary>
    public sealed record SelectionNode(PlanNode Input, Func<object[], bool> Predicate) : PlanNode;

    /// <summary>
    /// Projects each tuple to a new shape.
    /// </summary>
    public sealed record ProjectionNode(PlanNode Input, Func<object[], object[]> Project) : PlanNode;

    /// <summary>
    /// Performs a nested-loop join between two inputs.
    /// </summary>
    public sealed record JoinNode(
        PlanNode Left,
        PlanNode Right,
        Func<object[], object[], bool> Predicate,
        Func<object[], object[], object[]> Project
    ) : PlanNode;

    /// <summary>
    /// Concatenates the results of multiple sources.
    /// </summary>
    public sealed record UnionNode(IReadOnlyList<PlanNode> Sources) : PlanNode;

    /// <summary>
    /// Deduplicates tuples using a supplied comparer.
    /// </summary>
    public sealed record DistinctNode(PlanNode Input, IEqualityComparer<object[]>? Comparer = null) : PlanNode;

    /// <summary>
    /// Represents a fixpoint evaluation consisting of base and recursive plans.
    /// </summary>
    public sealed record FixpointNode(
        PlanNode BasePlan,
        IReadOnlyList<PlanNode> RecursivePlans,
        PredicateId Predicate
    ) : PlanNode;

    /// <summary>
    /// Indicates which relation a recursive reference should read from.
    /// </summary>
    public enum RecursiveRefKind
    {
        Total,
        Delta
    }

    /// <summary>
    /// References the evolving total or delta relation during fixpoint execution.
    /// </summary>
    public sealed record RecursiveRefNode(PredicateId Predicate, RecursiveRefKind Kind) : PlanNode;

    /// <summary>
    /// Produces no tuples; used when a plan lacks base clauses.
    /// </summary>
    public sealed record EmptyNode(int Width) : PlanNode;

    /// <summary>
    /// Query metadata used by the engine.
    /// </summary>
    public sealed record QueryPlan(
        PredicateId Head,
        PlanNode Root,
        bool IsRecursive = false
    );

    /// <summary>
    /// Executes <see cref="QueryPlan"/> instances.
    /// Currently supports non-recursive plans; recursion-aware fixpoint
    /// iteration will be layered on later.
    /// </summary>
    public sealed class InMemoryRelationProvider : IRelationProvider
    {
        private readonly Dictionary<PredicateId, List<object[]>> _store = new();

        public void AddFact(PredicateId predicate, params object[] values)
        {
            if (values is null) throw new ArgumentNullException(nameof(values));
            StoreFact(predicate, values);
        }

        public void AddFacts(PredicateId predicate, IEnumerable<object[]> tuples)
        {
            if (tuples is null) throw new ArgumentNullException(nameof(tuples));
            foreach (var tuple in tuples)
            {
                StoreFact(predicate, tuple);
            }
        }

        private void StoreFact(PredicateId predicate, object[] tuple)
        {
            if (tuple is null) throw new ArgumentNullException(nameof(tuple));
            if (!_store.TryGetValue(predicate, out var list))
            {
                list = new List<object[]>();
                _store[predicate] = list;
            }
            list.Add(tuple);
        }

        public IEnumerable<object[]> GetFacts(PredicateId predicate)
        {
            if (_store.TryGetValue(predicate, out var list))
            {
                return list;
            }
            return Array.Empty<object[]>();
        }
    }

    public sealed class QueryExecutor
    {
        private readonly IRelationProvider _provider;

        public QueryExecutor(IRelationProvider provider)
        {
            _provider = provider ?? throw new ArgumentNullException(nameof(provider));
        }

        public IEnumerable<object[]> Execute(QueryPlan plan)
        {
            if (plan is null) throw new ArgumentNullException(nameof(plan));
            return Evaluate(plan.Root);
        }

        private IEnumerable<object[]> Evaluate(PlanNode node, EvaluationContext? context = null)
        {
            switch (node)
            {
                case RelationScanNode scan:
                    return _provider.GetFacts(scan.Relation) ?? Enumerable.Empty<object[]>();

                case SelectionNode selection:
                    return Evaluate(selection.Input, context).Where(tuple => selection.Predicate(tuple));

                case ProjectionNode projection:
                    return Evaluate(projection.Input, context).Select(tuple => projection.Project(tuple));

                case JoinNode join:
                    return ExecuteJoin(join, context);

                case UnionNode union:
                    return ExecuteUnion(union, context);

                case DistinctNode distinct:
                    return ExecuteDistinct(distinct, context);

                case FixpointNode fixpoint:
                    return ExecuteFixpoint(fixpoint);

                case RecursiveRefNode recursiveRef:
                    return EvaluateRecursiveReference(recursiveRef, context);

                case EmptyNode:
                    return Enumerable.Empty<object[]>();

                default:
                    throw new NotSupportedException($"Unsupported plan node: {node.GetType().Name}");
            }
        }

        private IEnumerable<object[]> ExecuteJoin(JoinNode join, EvaluationContext? context)
        {
            var left = Evaluate(join.Left, context);
            var rightMaterialised = Evaluate(join.Right, context).ToList();

            foreach (var leftTuple in left)
            {
                foreach (var rightTuple in rightMaterialised)
                {
                    if (join.Predicate(leftTuple, rightTuple))
                    {
                        yield return join.Project(leftTuple, rightTuple);
                    }
                }
            }
        }

        private IEnumerable<object[]> ExecuteUnion(UnionNode union, EvaluationContext? context) =>
            union.Sources.SelectMany(node => Evaluate(node, context));

        private IEnumerable<object[]> ExecuteDistinct(DistinctNode distinct, EvaluationContext? context)
        {
            var comparer = distinct.Comparer ?? StructuralArrayComparer.Instance;
            var seen = new HashSet<RowWrapper>(new RowWrapperComparer(comparer));

            foreach (var tuple in Evaluate(distinct.Input, context))
            {
                if (seen.Add(new RowWrapper(tuple)))
                {
                    yield return tuple;
                }
            }
        }

        private IEnumerable<object[]> ExecuteFixpoint(FixpointNode fixpoint)
        {
            if (fixpoint is null) throw new ArgumentNullException(nameof(fixpoint));

            var comparer = StructuralArrayComparer.Instance;
            var totalSet = new HashSet<RowWrapper>(new RowWrapperComparer(comparer));
            var totalRows = new List<object[]>();

            // Seed with base clauses
            var baseRows = Evaluate(fixpoint.BasePlan).ToList();
            var deltaRows = new List<object[]>();
            foreach (var tuple in baseRows)
            {
                if (TryAddRow(totalSet, tuple))
                {
                    totalRows.Add(tuple);
                    deltaRows.Add(tuple);
                }
            }

            var context = new EvaluationContext(fixpoint.Predicate)
            {
                Total = totalRows,
                Delta = deltaRows
            };

            while (context.Delta.Count > 0)
            {
                var nextDelta = new List<object[]>();
                foreach (var recursivePlan in fixpoint.RecursivePlans)
                {
                    foreach (var tuple in Evaluate(recursivePlan, context))
                    {
                        if (TryAddRow(totalSet, tuple))
                        {
                            totalRows.Add(tuple);
                            nextDelta.Add(tuple);
                        }
                    }
                }
                context.Delta = nextDelta;
            }

            return totalRows;
        }

        private IEnumerable<object[]> EvaluateRecursiveReference(RecursiveRefNode node, EvaluationContext? context)
        {
            if (context is null)
            {
                throw new InvalidOperationException("Recursive reference evaluated without an execution context.");
            }

            if (!node.Predicate.Equals(context.Head))
            {
                throw new NotSupportedException($"Cross-predicate recursion is not supported (referenced {node.Predicate} within {context.Head}).");
            }

            return node.Kind switch
            {
                RecursiveRefKind.Total => context.Total,
                RecursiveRefKind.Delta => context.Delta,
                _ => throw new ArgumentOutOfRangeException(nameof(node.Kind), node.Kind, "Unknown recursive reference kind.")
            };
        }

        private static bool TryAddRow(HashSet<RowWrapper> set, object[] tuple)
        {
            if (tuple is null) throw new ArgumentNullException(nameof(tuple));
            return set.Add(new RowWrapper(tuple));
        }

        private sealed class EvaluationContext
        {
            public EvaluationContext(PredicateId head)
            {
                Head = head;
            }

            public PredicateId Head { get; }
            public IReadOnlyList<object[]> Total { get; set; } = Array.Empty<object[]>();
            public IReadOnlyList<object[]> Delta { get; set; } = Array.Empty<object[]>();
        }

        private sealed record RowWrapper(object[] Row);

        private sealed class RowWrapperComparer : IEqualityComparer<RowWrapper>
        {
            private readonly IEqualityComparer<object[]> _inner;

            public RowWrapperComparer(IEqualityComparer<object[]> inner)
            {
                _inner = inner;
            }

            public bool Equals(RowWrapper? x, RowWrapper? y)
            {
                if (ReferenceEquals(x, y)) return true;
                if (x is null || y is null) return false;
                return _inner.Equals(x.Row, y.Row);
            }

            public int GetHashCode(RowWrapper obj) => _inner.GetHashCode(obj.Row);
        }

        private sealed class StructuralArrayComparer : IEqualityComparer<object[]>
        {
            public static readonly StructuralArrayComparer Instance = new();

            public bool Equals(object[]? x, object[]? y)
            {
                if (ReferenceEquals(x, y)) return true;
                if (x is null || y is null) return false;
                if (x.Length != y.Length) return false;
                for (var i = 0; i < x.Length; i++)
                {
                    if (!Equals(x[i], y[i]))
                    {
                        return false;
                    }
                }
                return true;
            }

            public int GetHashCode(object[] obj)
            {
                unchecked
                {
                    var hash = 17;
                    foreach (var value in obj)
                    {
                        hash = hash * 31 + (value?.GetHashCode() ?? 0);
                    }
                    return hash;
                }
            }
        }
    }
}
