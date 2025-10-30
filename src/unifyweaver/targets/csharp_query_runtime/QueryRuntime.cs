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
            if (!_store.TryGetValue(predicate, out var list))
            {
                list = new List<object[]>();
                _store[predicate] = list;
            }
            list.Add(values);
        }

        public void AddFacts(PredicateId predicate, IEnumerable<object[]> tuples)
        {
            if (tuples is null) throw new ArgumentNullException(nameof(tuples));
            foreach (var tuple in tuples)
            {
                AddFact(predicate, tuple);
            }
        }

        private void AddFact(PredicateId predicate, object[] tuple)
        {
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
            if (plan.IsRecursive)
            {
                throw new NotSupportedException("Recursive plans are not yet supported by QueryExecutor.");
            }

            return Evaluate(plan.Root);
        }

        private IEnumerable<object[]> Evaluate(PlanNode node)
        {
            switch (node)
            {
                case RelationScanNode scan:
                    return _provider.GetFacts(scan.Relation) ?? Enumerable.Empty<object[]>();

                case SelectionNode selection:
                    return Evaluate(selection.Input).Where(tuple => selection.Predicate(tuple));

                case ProjectionNode projection:
                    return Evaluate(projection.Input).Select(tuple => projection.Project(tuple));

                case JoinNode join:
                    return ExecuteJoin(join);

                case UnionNode union:
                    return ExecuteUnion(union);

                case DistinctNode distinct:
                    return ExecuteDistinct(distinct);

                default:
                    throw new NotSupportedException($"Unsupported plan node: {node.GetType().Name}");
            }
        }

        private IEnumerable<object[]> ExecuteJoin(JoinNode join)
        {
            var left = Evaluate(join.Left);
            var rightMaterialised = Evaluate(join.Right).ToList();

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

        private IEnumerable<object[]> ExecuteUnion(UnionNode union) =>
            union.Sources.SelectMany(Evaluate);

        private IEnumerable<object[]> ExecuteDistinct(DistinctNode distinct)
        {
            var comparer = distinct.Comparer ?? StructuralArrayComparer.Instance;
            var seen = new HashSet<RowWrapper>(new RowWrapperComparer(comparer));

            foreach (var tuple in Evaluate(distinct.Input))
            {
                if (seen.Add(new RowWrapper(tuple)))
                {
                    yield return tuple;
                }
            }
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
