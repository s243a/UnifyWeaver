// SPDX-License-Identifier: MIT OR Apache-2.0
// Generated runtime scaffolding for UnifyWeaver C# query execution
// Provides minimal infrastructure for executing declarative plans
// emitted by the forthcoming csharp_query target.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using System.Xml.Linq;

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
    /// Represents a wildcard when matching tuples against a pattern.
    /// </summary>
    public static class Wildcard
    {
        public static readonly object Value = new();
    }

    public enum AggregateOperation
    {
        Count,
        Sum,
        Avg,
        Min,
        Max,
        Set,
        Bag
    }

    /// <summary>
    /// Seeds execution with caller-supplied parameter tuples.
    /// </summary>
    public sealed record ParamSeedNode(
        PredicateId Predicate,
        IReadOnlyList<int> InputPositions,
        int Width
    ) : PlanNode;

    /// <summary>
    /// Evaluates a subplan once and caches its tuples for reuse.
    /// </summary>
    public sealed record MaterializeNode(
        string Id,
        PlanNode Plan,
        int Width
    ) : PlanNode;

    /// <summary>
    /// Executes a set of definition nodes (typically materialising derived relations)
    /// before evaluating the main query body.
    /// </summary>
    public sealed record ProgramNode(
        IReadOnlyList<PlanNode> Definitions,
        PlanNode Body
    ) : PlanNode;

    /// <summary>
    /// Evaluates a plan and registers its output tuples as the facts for a predicate.
    /// </summary>
    public sealed record DefineRelationNode(
        PredicateId Predicate,
        PlanNode Plan
    ) : PlanNode;

    /// <summary>
    /// Scans a base relation provided by <see cref="IRelationProvider"/>.
    /// </summary>
    public sealed record RelationScanNode(PredicateId Relation) : PlanNode;

    /// <summary>
    /// Scans a relation using a fixed pattern (wildcards and constant values).
    /// </summary>
    public sealed record PatternScanNode(PredicateId Relation, object[] Pattern) : PlanNode;

    /// <summary>
    /// Applies a tuple-level filter.
    /// </summary>
    public sealed record SelectionNode(PlanNode Input, Func<object[], bool> Predicate) : PlanNode;

    /// <summary>
    /// Filters tuples by requiring that a bound negated predicate has no matching fact.
    /// </summary>
    public sealed record NegationNode(
        PlanNode Input,
        PredicateId Predicate,
        Func<object[], object[]> KeySelector
    ) : PlanNode;

    /// <summary>
    /// Applies an aggregate_all-style operation to a base relation, correlated with the input tuples.
    /// </summary>
    public sealed record AggregateNode(
        PlanNode Input,
        PredicateId Predicate,
        AggregateOperation Operation,
        Func<object[], object[]> Pattern,
        IReadOnlyList<int> GroupByIndices,
        int ValueIndex,
        int Width
    ) : PlanNode;

    /// <summary>
    /// Applies an aggregate operation to the results of a subplan, correlated with the input tuples.
    /// </summary>
    public sealed record AggregateSubplanNode(
        PlanNode Input,
        PlanNode Subplan,
        AggregateOperation Operation,
        Func<object[], object[]> ParameterSelector,
        IReadOnlyList<int> GroupByIndices,
        int ValueIndex,
        int Width
    ) : PlanNode;

    /// <summary>
    /// Projects each tuple to a new shape.
    /// </summary>
    public sealed record ProjectionNode(PlanNode Input, Func<object[], object[]> Project) : PlanNode;

    /// <summary>
    /// Performs a key-based equi-join between two inputs.
    /// </summary>
    public sealed record KeyJoinNode(
        PlanNode Left,
        PlanNode Right,
        IReadOnlyList<int> LeftKeys,
        IReadOnlyList<int> RightKeys,
        int LeftWidth,
        int RightWidth,
        int Width
    ) : PlanNode;

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

    public enum OrderDirection
    {
        Asc,
        Desc
    }

    public sealed record OrderKey(int Index, OrderDirection Direction = OrderDirection.Asc);

    /// <summary>
    /// Sorts tuples deterministically using a sequence of key columns.
    /// </summary>
    public sealed record OrderByNode(PlanNode Input, IReadOnlyList<OrderKey> Keys) : PlanNode;

    /// <summary>
    /// Limits output to the first N tuples.
    /// </summary>
    public sealed record LimitNode(PlanNode Input, int Count) : PlanNode;

    /// <summary>
    /// Skips the first N tuples.
    /// </summary>
    public sealed record OffsetNode(PlanNode Input, int Count) : PlanNode;

    /// <summary>
    /// Represents a fixpoint evaluation consisting of base and recursive plans.
    /// </summary>
    public sealed record FixpointNode(
        PlanNode BasePlan,
        IReadOnlyList<PlanNode> RecursivePlans,
        PredicateId Predicate
    ) : PlanNode;

    /// <summary>
    /// Computes the transitive closure of a binary edge relation.
    /// </summary>
    public sealed record TransitiveClosureNode(
        PredicateId EdgeRelation,
        PredicateId Predicate
    ) : PlanNode;

    /// <summary>
    /// Computes the transitive closure of an edge relation while preserving invariant group columns
    /// (e.g. label/category keyed reachability).
    /// </summary>
    public sealed record GroupedTransitiveClosureNode(
        PredicateId EdgeRelation,
        PredicateId Predicate,
        IReadOnlyList<int> GroupIndices
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
    /// References another predicate participating in a mutual fixpoint.
    /// </summary>
    public sealed record CrossRefNode(PredicateId Predicate, RecursiveRefKind Kind) : PlanNode;

    /// <summary>
    /// Represents the base and recursive plans for a predicate inside a mutual fixpoint.
    /// </summary>
    public sealed record MutualMember(PredicateId Predicate, PlanNode BasePlan, IReadOnlyList<PlanNode> RecursivePlans);

    /// <summary>
    /// Executes a set of predicates that depend on one another recursively.
    /// </summary>
    public sealed record MutualFixpointNode(IReadOnlyList<MutualMember> Members, PredicateId Head) : PlanNode;

    /// <summary>
    /// Evaluates a mutual fixpoint plan and registers the totals for all members as facts.
    /// </summary>
    public sealed record DefineMutualFixpointNode(MutualFixpointNode Fixpoint) : PlanNode;

    /// <summary>
    /// Base type for arithmetic expressions.
    /// </summary>
    public abstract record ArithmeticExpression;

    public sealed record ColumnExpression(int Index) : ArithmeticExpression;

    public sealed record ConstantExpression(object Value) : ArithmeticExpression;

    public sealed record UnaryArithmeticExpression(ArithmeticUnaryOperator Operator, ArithmeticExpression Operand) : ArithmeticExpression;

    public sealed record BinaryArithmeticExpression(ArithmeticBinaryOperator Operator, ArithmeticExpression Left, ArithmeticExpression Right) : ArithmeticExpression;

    public enum ArithmeticUnaryOperator
    {
        Negate
    }

    public enum ArithmeticBinaryOperator
    {
        Add,
        Subtract,
        Multiply,
        Divide,
        IntegerDivide,
        Modulo
    }

    /// <summary>
    /// Extends each tuple with the result of an arithmetic expression.
    /// </summary>
    public sealed record ArithmeticNode(
        PlanNode Input,
        ArithmeticExpression Expression,
        int ResultIndex,
        int Width
    ) : PlanNode;

    /// <summary>
    /// Produces no tuples; used when a plan lacks base clauses.
    /// </summary>
    public sealed record EmptyNode(int Width) : PlanNode;

    /// <summary>
    /// Produces exactly one tuple of the requested width.
    /// </summary>
    public sealed record UnitNode(int Width) : PlanNode;

    /// <summary>
    /// Query metadata used by the engine.
    /// </summary>
    public sealed record QueryPlan(
        PredicateId Head,
        PlanNode Root,
        bool IsRecursive = false,
        IReadOnlyList<int>? InputPositions = null
    );

    public sealed record QueryExecutorOptions(bool ReuseCaches = false);

    public sealed record QueryNodeTrace(
        int Id,
        string NodeType,
        long Invocations,
        long Enumerations,
        long Rows,
        TimeSpan Elapsed
    );

    public sealed record QueryCacheTrace(
        string Cache,
        string Key,
        long Lookups,
        long Hits,
        long Builds
    );

    public sealed record QueryStrategyTrace(
        int NodeId,
        string NodeType,
        string Strategy,
        long Count
    );

    public sealed record QueryFixpointIterationTrace(
        int NodeId,
        string NodeType,
        string Predicate,
        int Iteration,
        int DeltaRows,
        int TotalRows
    );

    public sealed class QueryExecutionTrace
    {
        private sealed class NodeStats
        {
            public int Id { get; init; }

            public long Invocations;

            public long Enumerations;

            public long Rows;

            public TimeSpan Elapsed;
        }

        private sealed class CacheStats
        {
            public long Lookups;

            public long Hits;

            public long Builds;
        }

        private sealed class PlanNodeStrategyComparer : IEqualityComparer<(PlanNode Node, string Strategy)>
        {
            public bool Equals((PlanNode Node, string Strategy) x, (PlanNode Node, string Strategy) y) =>
                ReferenceEquals(x.Node, y.Node) && string.Equals(x.Strategy, y.Strategy, StringComparison.Ordinal);

            public int GetHashCode((PlanNode Node, string Strategy) obj) =>
                HashCode.Combine(RuntimeHelpers.GetHashCode(obj.Node), obj.Strategy);
        }

        private sealed class ReferencePlanNodeComparer : IEqualityComparer<PlanNode>
        {
            public static readonly ReferencePlanNodeComparer Instance = new();

            public bool Equals(PlanNode? x, PlanNode? y) => ReferenceEquals(x, y);

            public int GetHashCode(PlanNode obj) => RuntimeHelpers.GetHashCode(obj);
        }

        private readonly Dictionary<PlanNode, NodeStats> _stats = new(ReferencePlanNodeComparer.Instance);
        private readonly Dictionary<(string Cache, string Key), CacheStats> _cacheStats = new();
        private readonly Dictionary<(PlanNode Node, string Strategy), long> _strategies = new(new PlanNodeStrategyComparer());
        private readonly List<QueryFixpointIterationTrace> _fixpointIterations = new();
        private int _nextId = 1;

        private NodeStats GetOrAdd(PlanNode node)
        {
            if (!_stats.TryGetValue(node, out var stats))
            {
                stats = new NodeStats { Id = _nextId++ };
                _stats.Add(node, stats);
            }

            return stats;
        }

        private CacheStats GetOrAdd(string cache, string key)
        {
            var cacheKey = (cache, key);
            if (!_cacheStats.TryGetValue(cacheKey, out var stats))
            {
                stats = new CacheStats();
                _cacheStats.Add(cacheKey, stats);
            }

            return stats;
        }

        internal void RecordInvocation(PlanNode node)
        {
            if (node is null) throw new ArgumentNullException(nameof(node));
            GetOrAdd(node).Invocations++;
        }

        internal void RecordCacheLookup(string cache, string key, bool hit, bool built)
        {
            if (cache is null) throw new ArgumentNullException(nameof(cache));
            if (key is null) throw new ArgumentNullException(nameof(key));

            var stats = GetOrAdd(cache, key);
            stats.Lookups++;
            if (hit)
            {
                stats.Hits++;
            }

            if (built)
            {
                stats.Builds++;
            }
        }

        internal void RecordStrategy(PlanNode node, string strategy)
        {
            if (node is null) throw new ArgumentNullException(nameof(node));
            if (strategy is null) throw new ArgumentNullException(nameof(strategy));

            _ = GetOrAdd(node);
            var key = (node, strategy);
            if (_strategies.TryGetValue(key, out var count))
            {
                _strategies[key] = count + 1;
            }
            else
            {
                _strategies[key] = 1;
            }
        }

        internal void RecordFixpointIteration(
            PlanNode node,
            PredicateId predicate,
            int iteration,
            int deltaRows,
            int totalRows)
        {
            if (node is null) throw new ArgumentNullException(nameof(node));

            var stats = GetOrAdd(node);
            _fixpointIterations.Add(new QueryFixpointIterationTrace(
                stats.Id,
                node.GetType().Name,
                predicate.ToString(),
                iteration,
                deltaRows,
                totalRows));
        }

        internal IEnumerable<object[]> WrapEnumeration(PlanNode node, IEnumerable<object[]> source)
        {
            if (node is null) throw new ArgumentNullException(nameof(node));
            if (source is null) throw new ArgumentNullException(nameof(source));

            var stats = GetOrAdd(node);

            return Iterator();

            IEnumerable<object[]> Iterator()
            {
                stats.Enumerations++;
                var stopwatch = Stopwatch.StartNew();
                long rows = 0;
                try
                {
                    foreach (var row in source)
                    {
                        rows++;
                        yield return row;
                    }
                }
                finally
                {
                    stopwatch.Stop();
                    stats.Rows += rows;
                    stats.Elapsed += stopwatch.Elapsed;
                }
            }
        }

        public IReadOnlyList<QueryNodeTrace> Snapshot()
        {
            return _stats
                .Select(kvp =>
                {
                    var stats = kvp.Value;
                    return new QueryNodeTrace(
                        stats.Id,
                        kvp.Key.GetType().Name,
                        stats.Invocations,
                        stats.Enumerations,
                        stats.Rows,
                        stats.Elapsed);
                })
                .OrderBy(s => s.Id)
                .ToList();
        }

        public IReadOnlyList<QueryCacheTrace> SnapshotCaches()
        {
            return _cacheStats
                .Select(kvp =>
                {
                    var stats = kvp.Value;
                    return new QueryCacheTrace(
                        kvp.Key.Cache,
                        kvp.Key.Key,
                        stats.Lookups,
                        stats.Hits,
                        stats.Builds);
                })
                .OrderBy(s => s.Cache, StringComparer.Ordinal)
                .ThenBy(s => s.Key, StringComparer.Ordinal)
                .ToList();
        }

        public IReadOnlyList<QueryStrategyTrace> SnapshotStrategies()
        {
            return _strategies
                .Select(kvp =>
                {
                    var node = kvp.Key.Node;
                    return new QueryStrategyTrace(
                        GetOrAdd(node).Id,
                        node.GetType().Name,
                        kvp.Key.Strategy,
                        kvp.Value);
                })
                .OrderBy(s => s.NodeId)
                .ThenBy(s => s.Strategy, StringComparer.Ordinal)
                .ToList();
        }

        public IReadOnlyList<QueryFixpointIterationTrace> SnapshotFixpointIterations()
        {
            return _fixpointIterations
                .OrderBy(s => s.NodeId)
                .ThenBy(s => s.Predicate, StringComparer.Ordinal)
                .ThenBy(s => s.Iteration)
                .ToList();
        }

        public override string ToString()
        {
            var builder = new StringBuilder();
            var nodes = Snapshot();
            for (var i = 0; i < nodes.Count; i++)
            {
                var s = nodes[i];
                builder.Append('[')
                    .Append(s.Id)
                    .Append("] ")
                    .Append(s.NodeType)
                    .Append(" invocations=")
                    .Append(s.Invocations)
                    .Append(" enumerations=")
                    .Append(s.Enumerations)
                    .Append(" rows=")
                    .Append(s.Rows)
                    .Append(" elapsed=")
                    .Append(s.Elapsed);
                if (i + 1 < nodes.Count)
                {
                    builder.AppendLine();
                }
            }

            var fixpoints = SnapshotFixpointIterations();
            if (fixpoints.Count > 0)
            {
                builder.AppendLine()
                    .AppendLine()
                    .AppendLine("Fixpoints:");

                foreach (var fp in fixpoints)
                {
                    builder.Append('[')
                        .Append(fp.NodeId)
                        .Append("] ")
                        .Append(fp.NodeType)
                        .Append(" predicate=")
                        .Append(fp.Predicate)
                        .Append(" iter=")
                        .Append(fp.Iteration)
                        .Append(" delta=")
                        .Append(fp.DeltaRows)
                        .Append(" total=")
                        .Append(fp.TotalRows)
                        .AppendLine();
                }
            }

            var caches = SnapshotCaches();
            if (caches.Count > 0)
            {
                builder.AppendLine()
                    .AppendLine()
                    .AppendLine("Caches:");

                foreach (var cache in caches)
                {
                    builder.Append(cache.Cache)
                        .Append(' ')
                        .Append(cache.Key)
                        .Append(" lookups=")
                        .Append(cache.Lookups)
                        .Append(" hits=")
                        .Append(cache.Hits)
                        .Append(" builds=")
                        .Append(cache.Builds)
                        .AppendLine();
                }
            }

            var strategies = SnapshotStrategies();
            if (strategies.Count > 0)
            {
                builder.AppendLine()
                    .AppendLine("Strategies:");

                foreach (var strategy in strategies)
                {
                    builder.Append('[')
                        .Append(strategy.NodeId)
                        .Append("] ")
                        .Append(strategy.NodeType)
                        .Append(' ')
                        .Append(strategy.Strategy)
                        .Append(" count=")
                        .Append(strategy.Count)
                        .AppendLine();
                }
            }

            return builder.ToString().TrimEnd();
        }
    }

    public static class QueryPlanExplainer
    {
        private sealed class ReferencePlanNodeComparer : IEqualityComparer<PlanNode>
        {
            public static readonly ReferencePlanNodeComparer Instance = new();

            public bool Equals(PlanNode? x, PlanNode? y) => ReferenceEquals(x, y);

            public int GetHashCode(PlanNode obj) => RuntimeHelpers.GetHashCode(obj);
        }

        public static string Explain(QueryPlan plan)
        {
            if (plan is null) throw new ArgumentNullException(nameof(plan));

            var builder = new StringBuilder();
            builder.AppendLine($"QueryPlan head={plan.Head} recursive={plan.IsRecursive}");
            if (plan.InputPositions is { Count: > 0 })
            {
                builder.AppendLine($"InputPositions=[{string.Join(",", plan.InputPositions)}]");
            }

            var ids = new Dictionary<PlanNode, int>(ReferencePlanNodeComparer.Instance);
            var expanded = new HashSet<PlanNode>(ReferencePlanNodeComparer.Instance);
            var nextId = 1;

            void WriteNode(string indent, string label, PlanNode node)
            {
                if (!ids.TryGetValue(node, out var id))
                {
                    id = nextId++;
                    ids[node] = id;
                }

                var alreadyExpanded = expanded.Contains(node);
                builder.Append(indent)
                    .Append(label)
                    .Append(": [")
                    .Append(id)
                    .Append("] ")
                    .Append(DescribeNode(node));
                if (alreadyExpanded)
                {
                    builder.AppendLine(" (ref)");
                    return;
                }

                builder.AppendLine();
                expanded.Add(node);

                var childIndent = indent + "  ";
                switch (node)
                {
                    case ParamSeedNode:
                    case RelationScanNode:
                    case PatternScanNode:
                    case RecursiveRefNode:
                    case CrossRefNode:
                    case EmptyNode:
                    case UnitNode:
                    case TransitiveClosureNode:
                    case GroupedTransitiveClosureNode:
                        return;

                    case ProgramNode program:
                        for (var i = 0; i < program.Definitions.Count; i++)
                        {
                            WriteNode(childIndent, $"define[{i}]", program.Definitions[i]);
                        }
                        WriteNode(childIndent, "body", program.Body);
                        return;

                    case DefineRelationNode defineRelation:
                        WriteNode(childIndent, "plan", defineRelation.Plan);
                        return;

                    case SelectionNode selection:
                        WriteNode(childIndent, "input", selection.Input);
                        return;

                    case NegationNode negation:
                        WriteNode(childIndent, "input", negation.Input);
                        return;

                    case AggregateNode aggregate:
                        WriteNode(childIndent, "input", aggregate.Input);
                        return;

                    case ProjectionNode projection:
                        WriteNode(childIndent, "input", projection.Input);
                        return;

                    case MaterializeNode materialize:
                        WriteNode(childIndent, "plan", materialize.Plan);
                        return;

                    case KeyJoinNode keyJoin:
                        WriteNode(childIndent, "left", keyJoin.Left);
                        WriteNode(childIndent, "right", keyJoin.Right);
                        return;

                    case JoinNode join:
                        WriteNode(childIndent, "left", join.Left);
                        WriteNode(childIndent, "right", join.Right);
                        return;

                    case UnionNode union:
                        for (var i = 0; i < union.Sources.Count; i++)
                        {
                            WriteNode(childIndent, $"source[{i}]", union.Sources[i]);
                        }
                        return;

                    case DistinctNode distinct:
                        WriteNode(childIndent, "input", distinct.Input);
                        return;

                    case OrderByNode orderBy:
                        WriteNode(childIndent, "input", orderBy.Input);
                        return;

                    case LimitNode limit:
                        WriteNode(childIndent, "input", limit.Input);
                        return;

                    case OffsetNode offset:
                        WriteNode(childIndent, "input", offset.Input);
                        return;

                    case FixpointNode fixpoint:
                        WriteNode(childIndent, "base", fixpoint.BasePlan);
                        for (var i = 0; i < fixpoint.RecursivePlans.Count; i++)
                        {
                            WriteNode(childIndent, $"recursive[{i}]", fixpoint.RecursivePlans[i]);
                        }
                        return;

                    case MutualFixpointNode mutual:
                        for (var i = 0; i < mutual.Members.Count; i++)
                        {
                            var member = mutual.Members[i];
                            builder.AppendLine($"{childIndent}member[{i}] predicate={member.Predicate}");
                            WriteNode(childIndent + "  ", "base", member.BasePlan);
                            for (var j = 0; j < member.RecursivePlans.Count; j++)
                            {
                                WriteNode(childIndent + "  ", $"recursive[{j}]", member.RecursivePlans[j]);
                            }
                        }
                        return;

                    case DefineMutualFixpointNode defineMutual:
                        WriteNode(childIndent, "fixpoint", defineMutual.Fixpoint);
                        return;

                    case AggregateSubplanNode aggregateSubplan:
                        WriteNode(childIndent, "input", aggregateSubplan.Input);
                        WriteNode(childIndent, "subplan", aggregateSubplan.Subplan);
                        return;

                    case ArithmeticNode arithmetic:
                        WriteNode(childIndent, "input", arithmetic.Input);
                        return;

                    default:
                        builder.AppendLine($"{childIndent}(children omitted for unsupported node type)");
                        return;
                }
            }

            WriteNode("", "root", plan.Root);
            return builder.ToString();
        }

        private static string DescribeNode(PlanNode node) =>
            node switch
            {
                ParamSeedNode seed => $"ParamSeed predicate={seed.Predicate} width={seed.Width} inputs=[{string.Join(",", seed.InputPositions)}]",
                MaterializeNode materialize => $"Materialize id=\"{materialize.Id}\" width={materialize.Width}",
                ProgramNode program => $"Program definitions={program.Definitions.Count}",
                DefineRelationNode defineRelation => $"DefineRelation predicate={defineRelation.Predicate}",
                DefineMutualFixpointNode defineMutual => $"DefineMutualFixpoint members={defineMutual.Fixpoint.Members.Count} head={defineMutual.Fixpoint.Head}",
                RelationScanNode scan => $"RelationScan relation={scan.Relation}",
                PatternScanNode scan => $"PatternScan relation={scan.Relation} pattern=[{string.Join(",", scan.Pattern.Select(FormatPatternValue))}]",
                SelectionNode => "Selection",
                NegationNode negation => $"Negation predicate={negation.Predicate}",
                AggregateNode aggregate => $"Aggregate predicate={aggregate.Predicate} op={aggregate.Operation} groupBy=[{string.Join(",", aggregate.GroupByIndices)}] valueIndex={aggregate.ValueIndex} width={aggregate.Width}",
                AggregateSubplanNode aggregateSubplan => $"AggregateSubplan op={aggregateSubplan.Operation} groupBy=[{string.Join(",", aggregateSubplan.GroupByIndices)}] valueIndex={aggregateSubplan.ValueIndex} width={aggregateSubplan.Width}",
                ProjectionNode => "Projection",
                KeyJoinNode join => $"KeyJoin leftKeys=[{string.Join(",", join.LeftKeys)}] rightKeys=[{string.Join(",", join.RightKeys)}] width={join.Width}",
                JoinNode => "Join",
                UnionNode union => $"Union sources={union.Sources.Count}",
                DistinctNode distinct => $"Distinct comparer={(distinct.Comparer?.GetType().Name ?? "default")}",
                OrderByNode orderBy => $"OrderBy keys=[{string.Join(",", orderBy.Keys.Select(k => $"{k.Index}:{k.Direction}"))}]",
                LimitNode limit => $"Limit count={limit.Count}",
                OffsetNode offset => $"Offset count={offset.Count}",
                TransitiveClosureNode closure => $"TransitiveClosure edge={closure.EdgeRelation}",
                GroupedTransitiveClosureNode closure => $"GroupedTransitiveClosure edge={closure.EdgeRelation} groups=[{string.Join(",", closure.GroupIndices)}]",
                FixpointNode fixpoint => $"Fixpoint predicate={fixpoint.Predicate} recursivePlans={fixpoint.RecursivePlans.Count}",
                MutualFixpointNode mutual => $"MutualFixpoint head={mutual.Head} members={mutual.Members.Count}",
                RecursiveRefNode recursiveRef => $"RecursiveRef predicate={recursiveRef.Predicate} kind={recursiveRef.Kind}",
                CrossRefNode crossRef => $"CrossRef predicate={crossRef.Predicate} kind={crossRef.Kind}",
                ArithmeticNode arithmetic => $"Arithmetic resultIndex={arithmetic.ResultIndex} width={arithmetic.Width} expr={DescribeArithmetic(arithmetic.Expression)}",
                EmptyNode empty => $"Empty width={empty.Width}",
                UnitNode unit => $"Unit width={unit.Width}",
                _ => node.GetType().Name
            };

        private static string DescribeArithmetic(ArithmeticExpression expression) =>
            expression switch
            {
                ColumnExpression col => $"col({col.Index})",
                ConstantExpression constant => $"const({FormatValue(constant.Value)})",
                UnaryArithmeticExpression unary => $"{unary.Operator}({DescribeArithmetic(unary.Operand)})",
                BinaryArithmeticExpression binary => $"({DescribeArithmetic(binary.Left)} {FormatOperator(binary.Operator)} {DescribeArithmetic(binary.Right)})",
                _ => expression.GetType().Name
            };

        private static string FormatValue(object value) =>
            value switch
            {
                null => "null",
                string s => $"\"{s}\"",
                _ => value.ToString() ?? string.Empty
            };

        private static string FormatPatternValue(object value) =>
            ReferenceEquals(value, Wildcard.Value)
                ? "*"
                : FormatValue(value);

        private static string FormatOperator(ArithmeticBinaryOperator op) =>
            op switch
            {
                ArithmeticBinaryOperator.Add => "+",
                ArithmeticBinaryOperator.Subtract => "-",
                ArithmeticBinaryOperator.Multiply => "*",
                ArithmeticBinaryOperator.Divide => "/",
                ArithmeticBinaryOperator.IntegerDivide => "//",
                ArithmeticBinaryOperator.Modulo => "mod",
                _ => op.ToString()
            };
    }

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
        private static readonly object NullFactIndexKey = new();
        private static readonly RowWrapperComparer StructuralRowWrapperComparer = new(StructuralArrayComparer.Instance);
        private readonly EvaluationContext? _cacheContext;

        public QueryExecutor(IRelationProvider provider, QueryExecutorOptions? options = null)
        {
            _provider = provider ?? throw new ArgumentNullException(nameof(provider));
            options ??= new QueryExecutorOptions();
            _cacheContext = options.ReuseCaches ? new EvaluationContext() : null;
        }

        public IEnumerable<object[]> Execute(
            QueryPlan plan,
            IEnumerable<object[]>? parameters = null,
            QueryExecutionTrace? trace = null,
            CancellationToken cancellationToken = default)
        {
            if (plan is null) throw new ArgumentNullException(nameof(plan));
            var paramList = parameters?.ToList() ?? new List<object[]>();
            var context = _cacheContext is null
                ? new EvaluationContext(paramList, trace: trace, cancellationToken: cancellationToken)
                : new EvaluationContext(paramList, parent: _cacheContext, trace: trace, cancellationToken: cancellationToken);
            var inputPositions = plan.InputPositions;
            if (inputPositions is { Count: > 0 })
            {
                if (paramList.Count == 0)
                {
                    return Enumerable.Empty<object[]>();
                }

                if (plan.Root is TransitiveClosureNode closure &&
                    inputPositions.Count == 1 &&
                    inputPositions[0] == 0)
                {
                    var rows = ExecuteSeededTransitiveClosure(closure, parameters: paramList, context);
                    var contextCancellationToken = context.CancellationToken;
                    if (contextCancellationToken.CanBeCanceled)
                    {
                        rows = WithCancellation(rows, contextCancellationToken);
                    }

                    var activeTrace = context.Trace;
                    activeTrace?.RecordInvocation(closure);
                    return activeTrace is null ? rows : activeTrace.WrapEnumeration(closure, rows);
                }

                if (plan.Root is TransitiveClosureNode closureByTarget &&
                    inputPositions.Count == 1 &&
                    inputPositions[0] == 1)
                {
                    var rows = ExecuteSeededTransitiveClosureByTarget(closureByTarget, parameters: paramList, context);
                    var contextCancellationToken = context.CancellationToken;
                    if (contextCancellationToken.CanBeCanceled)
                    {
                        rows = WithCancellation(rows, contextCancellationToken);
                    }

                    var activeTrace = context.Trace;
                    activeTrace?.RecordInvocation(closureByTarget);
                    return activeTrace is null ? rows : activeTrace.WrapEnumeration(closureByTarget, rows);
                }

                if (plan.Root is TransitiveClosureNode closurePairs &&
                    inputPositions.Count == 2 &&
                    inputPositions[0] == 0 &&
                    inputPositions[1] == 1)
                {
                    var rows = ExecuteSeededTransitiveClosurePairs(closurePairs, parameters: paramList, context);
                    var contextCancellationToken = context.CancellationToken;
                    if (contextCancellationToken.CanBeCanceled)
                    {
                        rows = WithCancellation(rows, contextCancellationToken);
                    }

                    var activeTrace = context.Trace;
                    activeTrace?.RecordInvocation(closurePairs);
                    return activeTrace is null ? rows : activeTrace.WrapEnumeration(closurePairs, rows);
                }

                if (plan.Root is GroupedTransitiveClosureNode groupedClosure)
                {
                    var rows = ExecuteSeededGroupedTransitiveClosure(groupedClosure, inputPositions, parameters: paramList, context);
                    var contextCancellationToken = context.CancellationToken;
                    if (contextCancellationToken.CanBeCanceled)
                    {
                        rows = WithCancellation(rows, contextCancellationToken);
                    }

                    var activeTrace = context.Trace;
                    activeTrace?.RecordInvocation(groupedClosure);
                    return activeTrace is null ? rows : activeTrace.WrapEnumeration(groupedClosure, rows);
                }

                if (plan.Root is RelationScanNode scan)
                {
                    var rows = ExecuteBoundFactScan(scan, inputPositions, paramList, context);
                    var activeTrace = context.Trace;
                    activeTrace?.RecordInvocation(scan);
                    return activeTrace is null ? rows : activeTrace.WrapEnumeration(scan, rows);
                }

                var filtered = FilterByParameters(Evaluate(plan.Root, context), inputPositions, paramList);
                return filtered;
            }

            return Evaluate(plan.Root, context);
        }

        public IAsyncEnumerable<object[]> ExecuteAsync(
            QueryPlan plan,
            IEnumerable<object[]>? parameters = null,
            QueryExecutionTrace? trace = null,
            CancellationToken cancellationToken = default)
        {
            if (plan is null) throw new ArgumentNullException(nameof(plan));
            return new ExecuteAsyncEnumerable(this, plan, parameters, trace, cancellationToken);
        }

        public void ClearCaches()
        {
            if (_cacheContext is null)
            {
                return;
            }

            _cacheContext.Facts.Clear();
            _cacheContext.FactSources.Clear();
            _cacheContext.FactSets.Clear();
            _cacheContext.FactIndices.Clear();
            _cacheContext.JoinIndices.Clear();
            _cacheContext.TransitiveClosureResults.Clear();
            _cacheContext.TransitiveClosureSeededResults.Clear();
            _cacheContext.TransitiveClosureSeededByTargetResults.Clear();
            _cacheContext.GroupedTransitiveClosureResults.Clear();
            _cacheContext.GroupedTransitiveClosureSeededResults.Clear();
            _cacheContext.GroupedTransitiveClosureSeededByTargetResults.Clear();
        }

        private IEnumerable<object[]> Evaluate(PlanNode node, EvaluationContext? context = null)
        {
            var trace = context?.Trace;
            trace?.RecordInvocation(node);

            IEnumerable<object[]> result;
            switch (node)
            {
                case ParamSeedNode seed:
                    result = EvaluateParamSeed(seed, context);
                    break;

                case MaterializeNode materialize:
                    result = EvaluateMaterialize(materialize, context);
                    break;

                case ProgramNode program:
                    result = ExecuteProgram(program, context);
                    break;

                case DefineRelationNode defineRelation:
                    result = ExecuteDefineRelation(defineRelation, context);
                    break;

                case RelationScanNode scan:
                    result = context is null
                        ? _provider.GetFacts(scan.Relation) ?? Enumerable.Empty<object[]>()
                        : GetFactsList(scan.Relation, context);
                    break;

                case PatternScanNode scan:
                    result = ExecutePatternScan(scan, context);
                    break;

                case SelectionNode selection:
                    result = Evaluate(selection.Input, context).Where(tuple => selection.Predicate(tuple));
                    break;

                case NegationNode negation:
                    result = ExecuteNegation(negation, context);
                    break;

                case AggregateNode aggregate:
                    result = ExecuteAggregate(aggregate, context);
                    break;

                case AggregateSubplanNode aggregateSubplan:
                    result = ExecuteAggregateSubplan(aggregateSubplan, context);
                    break;

                case ArithmeticNode arithmetic:
                    result = ExecuteArithmetic(arithmetic, context);
                    break;

                case ProjectionNode projection:
                    result = Evaluate(projection.Input, context).Select(tuple => projection.Project(tuple));
                    break;

                case KeyJoinNode keyJoin:
                    result = ExecuteKeyJoin(keyJoin, context);
                    break;

                case JoinNode join:
                    result = ExecuteJoin(join, context);
                    break;

                case UnionNode union:
                    result = ExecuteUnion(union, context);
                    break;

                case DistinctNode distinct:
                    result = ExecuteDistinct(distinct, context);
                    break;

                case OrderByNode orderBy:
                    result = ExecuteOrderBy(orderBy, context);
                    break;

                case LimitNode limit:
                    result = ExecuteLimit(limit, context);
                    break;

                case OffsetNode offset:
                    result = ExecuteOffset(offset, context);
                    break;

                case TransitiveClosureNode closure:
                    result = ExecuteTransitiveClosure(closure, context);
                    break;

                case GroupedTransitiveClosureNode closure:
                    result = ExecuteGroupedTransitiveClosure(closure, context);
                    break;

                case FixpointNode fixpoint:
                    result = ExecuteFixpoint(fixpoint, context);
                    break;

                case MutualFixpointNode mutualFixpoint:
                    result = ExecuteMutualFixpoint(mutualFixpoint, context);
                    break;

                case DefineMutualFixpointNode defineMutual:
                    result = ExecuteDefineMutualFixpoint(defineMutual, context);
                    break;

                case RecursiveRefNode recursiveRef:
                    result = EvaluateRecursiveReference(recursiveRef, context);
                    break;

                case CrossRefNode crossRef:
                    result = EvaluateCrossReference(crossRef, context);
                    break;

                case EmptyNode:
                    result = Enumerable.Empty<object[]>();
                    break;

                case UnitNode unit:
                    result = EvaluateUnit(unit);
                    break;

                default:
                    throw new NotSupportedException($"Unsupported plan node: {node.GetType().Name}");
            }

            var cancellationToken = context?.CancellationToken ?? default;
            if (cancellationToken.CanBeCanceled)
            {
                result = WithCancellation(result, cancellationToken);
            }

            return trace is null ? result : trace.WrapEnumeration(node, result);
        }

        private static IEnumerable<object[]> WithCancellation(IEnumerable<object[]> source, CancellationToken cancellationToken)
        {
            if (source is null) throw new ArgumentNullException(nameof(source));
            if (!cancellationToken.CanBeCanceled)
            {
                return source;
            }

            return WithCancellationIterator(source, cancellationToken);
        }

        private static IEnumerable<object[]> WithCancellationIterator(IEnumerable<object[]> source, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var counter = 0;
            foreach (var row in source)
            {
                if ((++counter & 0x3FF) == 0)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                }

                yield return row;
            }

            cancellationToken.ThrowIfCancellationRequested();
        }

        private sealed class ExecuteAsyncEnumerable : IAsyncEnumerable<object[]>
        {
            private readonly QueryExecutor _executor;
            private readonly QueryPlan _plan;
            private readonly IEnumerable<object[]>? _parameters;
            private readonly QueryExecutionTrace? _trace;
            private readonly CancellationToken _cancellationToken;

            public ExecuteAsyncEnumerable(
                QueryExecutor executor,
                QueryPlan plan,
                IEnumerable<object[]>? parameters,
                QueryExecutionTrace? trace,
                CancellationToken cancellationToken)
            {
                _executor = executor ?? throw new ArgumentNullException(nameof(executor));
                _plan = plan ?? throw new ArgumentNullException(nameof(plan));
                _parameters = parameters;
                _trace = trace;
                _cancellationToken = cancellationToken;
            }

            public IAsyncEnumerator<object[]> GetAsyncEnumerator(CancellationToken cancellationToken = default)
            {
                var combinedToken = CombineCancellationTokens(_cancellationToken, cancellationToken, out var linkedCts);
                var enumerator = _executor.Execute(_plan, _parameters, _trace, combinedToken).GetEnumerator();
                return new ExecuteAsyncEnumerator(enumerator, combinedToken, linkedCts);
            }

            private static CancellationToken CombineCancellationTokens(
                CancellationToken first,
                CancellationToken second,
                out CancellationTokenSource? linkedCts)
            {
                linkedCts = null;

                if (!first.CanBeCanceled) return second;
                if (!second.CanBeCanceled) return first;
                if (first == second) return first;

                linkedCts = CancellationTokenSource.CreateLinkedTokenSource(first, second);
                return linkedCts.Token;
            }
        }

        private sealed class ExecuteAsyncEnumerator : IAsyncEnumerator<object[]>
        {
            private readonly IEnumerator<object[]> _inner;
            private readonly CancellationToken _cancellationToken;
            private readonly CancellationTokenSource? _linkedCts;
            private int _counter;

            public ExecuteAsyncEnumerator(
                IEnumerator<object[]> inner,
                CancellationToken cancellationToken,
                CancellationTokenSource? linkedCts)
            {
                _inner = inner ?? throw new ArgumentNullException(nameof(inner));
                _cancellationToken = cancellationToken;
                _linkedCts = linkedCts;
            }

            public object[] Current => _inner.Current;

            public ValueTask<bool> MoveNextAsync()
            {
                if (_cancellationToken.CanBeCanceled && ((_counter++ & 0x3FF) == 0))
                {
                    _cancellationToken.ThrowIfCancellationRequested();
                }

                return new ValueTask<bool>(_inner.MoveNext());
            }

            public ValueTask DisposeAsync()
            {
                _inner.Dispose();
                _linkedCts?.Dispose();
                return ValueTask.CompletedTask;
            }
        }

        private IEnumerable<object[]> ExecutePatternScan(PatternScanNode scan, EvaluationContext? context)
        {
            if (scan is null) throw new ArgumentNullException(nameof(scan));

            if (context is null)
            {
                return (_provider.GetFacts(scan.Relation) ?? Enumerable.Empty<object[]>()).Where(tuple =>
                    tuple is not null && TupleMatchesPattern(tuple, scan.Pattern));
            }

            var facts = GetFactsList(scan.Relation, context);
            var candidates = SelectFactsForPattern(scan.Relation, facts, scan.Pattern, context);
            return candidates.Where(tuple => tuple is not null && TupleMatchesPattern(tuple, scan.Pattern));
        }

        private static bool TupleMatchesPattern(object[] tuple, object[] pattern)
        {
            if (pattern is null) throw new ArgumentNullException(nameof(pattern));
            if (tuple is null) return false;

            var max = Math.Min(tuple.Length, pattern.Length);
            for (var i = 0; i < max; i++)
            {
                var expected = pattern[i];
                if (ReferenceEquals(expected, Wildcard.Value))
                {
                    continue;
                }

                if (!Equals(tuple[i], expected))
                {
                    return false;
                }
            }

            for (var i = max; i < pattern.Length; i++)
            {
                if (!ReferenceEquals(pattern[i], Wildcard.Value))
                {
                    return false;
                }
            }

            return true;
        }

        private IEnumerable<object[]> ExecuteProgram(ProgramNode program, EvaluationContext? context)
        {
            if (program is null) throw new ArgumentNullException(nameof(program));
            if (context is null) throw new InvalidOperationException("Program node evaluated without an execution context.");

            var allowReuse = _cacheContext is not null && context.FixpointDepth == 0;
            var volatilePredicates = allowReuse
                ? ComputeVolatilePredicates(program.Definitions)
                : null;

            foreach (var definition in program.Definitions)
            {
                if (allowReuse &&
                    volatilePredicates is not null &&
                    TryReuseDefinition(definition, context, volatilePredicates, out var cacheKey, out var reuseHit))
                {
                    context.Trace?.RecordCacheLookup("ProgramDefinition", cacheKey, hit: reuseHit, built: !reuseHit);
                    if (reuseHit)
                    {
                        continue;
                    }
                }

                _ = Evaluate(definition, context);
            }

            return Evaluate(program.Body, context);
        }

        private static bool TryReuseDefinition(
            PlanNode definition,
            EvaluationContext context,
            ISet<PredicateId> volatilePredicates,
            out string cacheKey,
            out bool reuseHit)
        {
            cacheKey = string.Empty;
            reuseHit = false;

            static bool HasCachedFacts(PredicateId predicate, PlanNode source, EvaluationContext context) =>
                context.FactSources.TryGetValue(predicate, out var cachedSource) &&
                ReferenceEquals(cachedSource, source) &&
                context.Facts.ContainsKey(predicate);

            switch (definition)
            {
                case DefineRelationNode defineRelation:
                    if (volatilePredicates.Contains(defineRelation.Predicate))
                    {
                        return false;
                    }

                    cacheKey = defineRelation.Predicate.ToString();
                    reuseHit = HasCachedFacts(defineRelation.Predicate, defineRelation, context);
                    return true;

                case DefineMutualFixpointNode defineMutual:
                    if (defineMutual.Fixpoint.Members.Any(member => volatilePredicates.Contains(member.Predicate)))
                    {
                        return false;
                    }

                    cacheKey = $"mutual:{defineMutual.Fixpoint.Head}";
                    reuseHit = defineMutual.Fixpoint.Members.All(member =>
                        HasCachedFacts(member.Predicate, defineMutual, context));
                    return true;

                default:
                    return false;
            }
        }

        private static bool ContainsParamSeed(PlanNode node) => node switch
        {
            ParamSeedNode => true,
            MaterializeNode materialize => ContainsParamSeed(materialize.Plan),
            ProgramNode program => program.Definitions.Any(ContainsParamSeed) || ContainsParamSeed(program.Body),
            DefineRelationNode define => ContainsParamSeed(define.Plan),
            SelectionNode selection => ContainsParamSeed(selection.Input),
            NegationNode negation => ContainsParamSeed(negation.Input),
            AggregateNode aggregate => ContainsParamSeed(aggregate.Input),
            AggregateSubplanNode aggregateSubplan => ContainsParamSeed(aggregateSubplan.Input) || ContainsParamSeed(aggregateSubplan.Subplan),
            ArithmeticNode arithmetic => ContainsParamSeed(arithmetic.Input),
            ProjectionNode projection => ContainsParamSeed(projection.Input),
            KeyJoinNode keyJoin => ContainsParamSeed(keyJoin.Left) || ContainsParamSeed(keyJoin.Right),
            JoinNode join => ContainsParamSeed(join.Left) || ContainsParamSeed(join.Right),
            UnionNode union => union.Sources.Any(ContainsParamSeed),
            DistinctNode distinct => ContainsParamSeed(distinct.Input),
            OrderByNode orderBy => ContainsParamSeed(orderBy.Input),
            LimitNode limit => ContainsParamSeed(limit.Input),
            OffsetNode offset => ContainsParamSeed(offset.Input),
            FixpointNode fixpoint => ContainsParamSeed(fixpoint.BasePlan) || fixpoint.RecursivePlans.Any(ContainsParamSeed),
            MutualFixpointNode mutual => mutual.Members.Any(member =>
                ContainsParamSeed(member.BasePlan) || member.RecursivePlans.Any(ContainsParamSeed)),
            DefineMutualFixpointNode defineMutual => ContainsParamSeed(defineMutual.Fixpoint),
            _ => false
        };

        private static HashSet<PredicateId> ComputeVolatilePredicates(IReadOnlyList<PlanNode> definitions)
        {
            var infos = new List<(IReadOnlyList<PredicateId> Defined, PlanNode Plan)>();

            foreach (var definition in definitions)
            {
                switch (definition)
                {
                    case DefineRelationNode defineRelation:
                        infos.Add((new[] { defineRelation.Predicate }, defineRelation.Plan));
                        break;

                    case DefineMutualFixpointNode defineMutual:
                        infos.Add((defineMutual.Fixpoint.Members.Select(member => member.Predicate).ToList(), defineMutual.Fixpoint));
                        break;
                }
            }

            var volatilePredicates = new HashSet<PredicateId>();
            foreach (var info in infos)
            {
                if (!ContainsParamSeed(info.Plan))
                {
                    continue;
                }

                foreach (var predicate in info.Defined)
                {
                    volatilePredicates.Add(predicate);
                }
            }

            if (volatilePredicates.Count == 0)
            {
                return volatilePredicates;
            }

            var referenced = new HashSet<PredicateId>();
            var changed = true;

            while (changed)
            {
                changed = false;
                foreach (var info in infos)
                {
                    if (info.Defined.Any(volatilePredicates.Contains))
                    {
                        continue;
                    }

                    referenced.Clear();
                    CollectReferencedPredicates(info.Plan, referenced);

                    if (!referenced.Any(volatilePredicates.Contains))
                    {
                        continue;
                    }

                    foreach (var predicate in info.Defined)
                    {
                        if (volatilePredicates.Add(predicate))
                        {
                            changed = true;
                        }
                    }
                }
            }

            return volatilePredicates;
        }

        private static void CollectReferencedPredicates(PlanNode node, ISet<PredicateId> predicates)
        {
            switch (node)
            {
                case RelationScanNode scan:
                    predicates.Add(scan.Relation);
                    return;

                case PatternScanNode scan:
                    predicates.Add(scan.Relation);
                    return;

                case ParamSeedNode seed:
                    predicates.Add(seed.Predicate);
                    return;

                case RecursiveRefNode recursive:
                    predicates.Add(recursive.Predicate);
                    return;

                case CrossRefNode cross:
                    predicates.Add(cross.Predicate);
                    return;

                case MaterializeNode materialize:
                    CollectReferencedPredicates(materialize.Plan, predicates);
                    return;

                case ProgramNode program:
                    foreach (var definition in program.Definitions)
                    {
                        CollectReferencedPredicates(definition, predicates);
                    }
                    CollectReferencedPredicates(program.Body, predicates);
                    return;

                case DefineRelationNode define:
                    CollectReferencedPredicates(define.Plan, predicates);
                    return;

                case DefineMutualFixpointNode defineMutual:
                    CollectReferencedPredicates(defineMutual.Fixpoint, predicates);
                    return;

                case FixpointNode fixpoint:
                    CollectReferencedPredicates(fixpoint.BasePlan, predicates);
                    foreach (var recursivePlan in fixpoint.RecursivePlans)
                    {
                        CollectReferencedPredicates(recursivePlan, predicates);
                    }
                    return;

                case TransitiveClosureNode closure:
                    predicates.Add(closure.EdgeRelation);
                    return;

                case MutualFixpointNode mutual:
                    foreach (var member in mutual.Members)
                    {
                        CollectReferencedPredicates(member.BasePlan, predicates);
                        foreach (var recursivePlan in member.RecursivePlans)
                        {
                            CollectReferencedPredicates(recursivePlan, predicates);
                        }
                    }
                    return;

                case SelectionNode selection:
                    CollectReferencedPredicates(selection.Input, predicates);
                    return;

                case NegationNode negation:
                    predicates.Add(negation.Predicate);
                    CollectReferencedPredicates(negation.Input, predicates);
                    return;

                case AggregateNode aggregate:
                    predicates.Add(aggregate.Predicate);
                    CollectReferencedPredicates(aggregate.Input, predicates);
                    return;

                case AggregateSubplanNode aggregateSubplan:
                    CollectReferencedPredicates(aggregateSubplan.Input, predicates);
                    CollectReferencedPredicates(aggregateSubplan.Subplan, predicates);
                    return;

                case ArithmeticNode arithmetic:
                    CollectReferencedPredicates(arithmetic.Input, predicates);
                    return;

                case ProjectionNode projection:
                    CollectReferencedPredicates(projection.Input, predicates);
                    return;

                case KeyJoinNode keyJoin:
                    CollectReferencedPredicates(keyJoin.Left, predicates);
                    CollectReferencedPredicates(keyJoin.Right, predicates);
                    return;

                case JoinNode join:
                    CollectReferencedPredicates(join.Left, predicates);
                    CollectReferencedPredicates(join.Right, predicates);
                    return;

                case UnionNode union:
                    foreach (var source in union.Sources)
                    {
                        CollectReferencedPredicates(source, predicates);
                    }
                    return;

                case DistinctNode distinct:
                    CollectReferencedPredicates(distinct.Input, predicates);
                    return;

                case OrderByNode orderBy:
                    CollectReferencedPredicates(orderBy.Input, predicates);
                    return;

                case LimitNode limit:
                    CollectReferencedPredicates(limit.Input, predicates);
                    return;

                case OffsetNode offset:
                    CollectReferencedPredicates(offset.Input, predicates);
                    return;
            }
        }

        private IEnumerable<object[]> ExecuteDefineRelation(DefineRelationNode define, EvaluationContext? context)
        {
            if (define is null) throw new ArgumentNullException(nameof(define));
            if (context is null) throw new InvalidOperationException("DefineRelation node evaluated without an execution context.");

            var evaluated = Evaluate(define.Plan, context);
            var rows = evaluated as List<object[]> ?? evaluated.ToList();
            RegisterFacts(define.Predicate, rows, context, define);
            return rows;
        }

        private IEnumerable<object[]> ExecuteDefineMutualFixpoint(DefineMutualFixpointNode define, EvaluationContext? context)
        {
            if (define is null) throw new ArgumentNullException(nameof(define));
            if (context is null) throw new InvalidOperationException("DefineMutualFixpoint node evaluated without an execution context.");

            var headRows = Evaluate(define.Fixpoint, context);

            foreach (var member in define.Fixpoint.Members)
            {
                if (context.Totals.TryGetValue(member.Predicate, out var totals))
                {
                    RegisterFacts(member.Predicate, totals, context, define);
                }
                else
                {
                    RegisterFacts(member.Predicate, new List<object[]>(), context, define);
                }
            }

            return headRows;
        }

        private static void RegisterFacts(PredicateId predicate, List<object[]> rows, EvaluationContext context, PlanNode? source)
        {
            context.Facts[predicate] = rows;
            if (source is null)
            {
                context.FactSources.Remove(predicate);
            }
            else
            {
                context.FactSources[predicate] = source;
            }
            context.FactSets.Remove(predicate);

            foreach (var key in context.FactIndices.Keys.Where(key => key.Predicate.Equals(predicate)).ToList())
            {
                context.FactIndices.Remove(key);
            }

            foreach (var key in context.JoinIndices.Keys.Where(key => key.Predicate.Equals(predicate)).ToList())
            {
                context.JoinIndices.Remove(key);
            }

            foreach (var key in context.RecursiveFactIndices.Keys.Where(key => key.Predicate.Equals(predicate)).ToList())
            {
                context.RecursiveFactIndices.Remove(key);
            }

            foreach (var key in context.RecursiveJoinIndices.Keys.Where(key => key.Predicate.Equals(predicate)).ToList())
            {
                context.RecursiveJoinIndices.Remove(key);
            }

            foreach (var key in context.TransitiveClosureResults.Keys
                         .Where(key => key.EdgeRelation.Equals(predicate) || key.Predicate.Equals(predicate))
                         .ToList())
            {
                context.TransitiveClosureResults.Remove(key);
            }

            foreach (var key in context.TransitiveClosureSeededResults.Keys
                         .Where(key => key.EdgeRelation.Equals(predicate) || key.Predicate.Equals(predicate))
                         .ToList())
            {
                context.TransitiveClosureSeededResults.Remove(key);
            }

            foreach (var key in context.TransitiveClosureSeededByTargetResults.Keys
                         .Where(key => key.EdgeRelation.Equals(predicate) || key.Predicate.Equals(predicate))
                         .ToList())
            {
                context.TransitiveClosureSeededByTargetResults.Remove(key);
            }

            foreach (var key in context.GroupedTransitiveClosureResults.Keys
                         .Where(key => key.EdgeRelation.Equals(predicate) || key.Predicate.Equals(predicate))
                         .ToList())
            {
                context.GroupedTransitiveClosureResults.Remove(key);
            }

            foreach (var key in context.GroupedTransitiveClosureSeededResults.Keys
                         .Where(key => key.EdgeRelation.Equals(predicate) || key.Predicate.Equals(predicate))
                         .ToList())
            {
                context.GroupedTransitiveClosureSeededResults.Remove(key);
            }

            foreach (var key in context.GroupedTransitiveClosureSeededByTargetResults.Keys
                         .Where(key => key.EdgeRelation.Equals(predicate) || key.Predicate.Equals(predicate))
                         .ToList())
            {
                context.GroupedTransitiveClosureSeededByTargetResults.Remove(key);
            }
        }

        private IEnumerable<object[]> ExecuteJoin(JoinNode join, EvaluationContext? context)
        {
            var left = Evaluate(join.Left, context);
            var rightRows = Evaluate(join.Right, context);
            var rightMaterialised = rightRows as List<object[]> ?? rightRows.ToList();

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

        private IEnumerable<object[]> ExecuteKeyJoin(KeyJoinNode join, EvaluationContext? context)
        {
            var trace = context?.Trace;

            if (join.LeftKeys is null || join.RightKeys is null || join.LeftKeys.Count == 0 || join.RightKeys.Count == 0)
            {
                trace?.RecordStrategy(join, "KeyJoinNestedLoop");
                var leftRows = Evaluate(join.Left, context);
                var rightSource = Evaluate(join.Right, context);
                var rightRows = rightSource as List<object[]> ?? rightSource.ToList();
                foreach (var leftTuple in leftRows)
                {
                    if (leftTuple is null) continue;

                    foreach (var rightTuple in rightRows)
                    {
                        if (rightTuple is null) continue;
                        yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                    }
                }
                yield break;
            }

            if (join.LeftKeys.Count != join.RightKeys.Count)
            {
                throw new InvalidOperationException($"KeyJoinNode expects equal key arity (left={join.LeftKeys.Count}, right={join.RightKeys.Count}).");
            }

            static object GetLookupKey(object[] tuple, int index, object nullKey)
            {
                object? value = null;
                if (index >= 0 && index < tuple.Length)
                {
                    value = tuple[index];
                }

                return value ?? nullKey;
            }

            var joinKeyCount = join.LeftKeys.Count;
            bool? forceBuildLeft = null;

            static bool TryGetPredicateScan(PlanNode node, out PredicateId predicate, out object[]? pattern, out Func<object[], bool>? filter)
            {
                predicate = default;
                pattern = null;
                filter = null;

                switch (node)
                {
                    case RelationScanNode scan:
                        predicate = scan.Relation;
                        return true;

                    case PatternScanNode scan:
                        predicate = scan.Relation;
                        pattern = scan.Pattern;
                        return true;

                    case SelectionNode selection:
                        if (TryGetPredicateScan(selection.Input, out predicate, out pattern, out var innerFilter))
                        {
                            filter = innerFilter is null
                                ? selection.Predicate
                                : tuple => innerFilter(tuple) && selection.Predicate(tuple);
                            return true;
                        }

                        return false;

                    default:
                        return false;
                }
            }

            static IReadOnlyList<int> GetScanIndexKeys(IReadOnlyList<int> joinKeys, object[]? pattern)
            {
                if (pattern is null)
                {
                    return joinKeys;
                }

                List<int>? indexKeys = null;

                for (var i = 0; i < pattern.Length; i++)
                {
                    var value = pattern[i];
                    if (ReferenceEquals(value, Wildcard.Value))
                    {
                        continue;
                    }

                    var isJoinKey = false;
                    for (var j = 0; j < joinKeys.Count; j++)
                    {
                        if (joinKeys[j] == i)
                        {
                            isJoinKey = true;
                            break;
                        }
                    }

                    if (isJoinKey)
                    {
                        continue;
                    }

                    if (indexKeys is null)
                    {
                        indexKeys = new List<int>(joinKeys.Count + 1);
                        for (var j = 0; j < joinKeys.Count; j++)
                        {
                            indexKeys.Add(joinKeys[j]);
                        }
                    }

                    indexKeys.Add(i);
                }

                return indexKeys ?? joinKeys;
            }

            static bool ProbeMatchesScanJoinKeyConstants(
                object[] probeTuple,
                IReadOnlyList<int> probeKeys,
                IReadOnlyList<int> scanKeys,
                object[]? scanPattern)
            {
                if (scanPattern is null)
                {
                    return true;
                }

                var keyCount = Math.Min(probeKeys.Count, scanKeys.Count);
                for (var i = 0; i < keyCount; i++)
                {
                    var scanIndex = scanKeys[i];
                    if (scanIndex < 0 || scanIndex >= scanPattern.Length)
                    {
                        continue;
                    }

                    var expected = scanPattern[scanIndex];
                    if (ReferenceEquals(expected, Wildcard.Value))
                    {
                        continue;
                    }

                    var probeIndex = probeKeys[i];
                    var actual = probeIndex >= 0 && probeIndex < probeTuple.Length ? probeTuple[probeIndex] : null;

                    if (!Equals(actual, expected))
                    {
                        return false;
                    }
                }

                return true;
            }

            static object[] BuildScanJoinLookupKey(
                object[] probeTuple,
                IReadOnlyList<int> probeKeys,
                IReadOnlyList<int> scanKeys,
                IReadOnlyList<int> scanIndexKeys,
                object[]? scanPattern)
            {
                var key = new object[scanIndexKeys.Count];

                for (var i = 0; i < scanIndexKeys.Count; i++)
                {
                    var scanIndex = scanIndexKeys[i];
                    var joinPos = -1;

                    for (var j = 0; j < scanKeys.Count; j++)
                    {
                        if (scanKeys[j] == scanIndex)
                        {
                            joinPos = j;
                            break;
                        }
                    }

                    if (joinPos >= 0)
                    {
                        var probeIndex = probeKeys[joinPos];
                        key[i] = probeIndex >= 0 && probeIndex < probeTuple.Length ? probeTuple[probeIndex] : null!;
                        continue;
                    }

                    if (scanPattern is not null && scanIndex >= 0 && scanIndex < scanPattern.Length)
                    {
                        key[i] = scanPattern[scanIndex];
                        continue;
                    }

                    key[i] = null!;
                }

                return key;
            }

            bool TryEstimateRowUpperBound(PlanNode node, EvaluationContext context, out int upperBound)
            {
                upperBound = 0;
                switch (node)
                {
                    case ParamSeedNode:
                        upperBound = context.Parameters.Count;
                        return true;

                    case UnitNode:
                        upperBound = 1;
                        return true;

                    case EmptyNode:
                        upperBound = 0;
                        return true;

                    case LimitNode limit:
                        upperBound = Math.Max(0, limit.Count);
                        return true;

                    case ProjectionNode projection:
                        return TryEstimateRowUpperBound(projection.Input, context, out upperBound);

                    case SelectionNode selection:
                        return TryEstimateRowUpperBound(selection.Input, context, out upperBound);

                    case ArithmeticNode arithmetic:
                        return TryEstimateRowUpperBound(arithmetic.Input, context, out upperBound);

                    case NegationNode negation:
                        return TryEstimateRowUpperBound(negation.Input, context, out upperBound);

                    case AggregateNode aggregate:
                        return TryEstimateRowUpperBound(aggregate.Input, context, out upperBound);

                    case AggregateSubplanNode aggregateSubplan:
                        return TryEstimateRowUpperBound(aggregateSubplan.Input, context, out upperBound);

                    case OrderByNode orderBy:
                        return TryEstimateRowUpperBound(orderBy.Input, context, out upperBound);

                    case DistinctNode distinct:
                        return TryEstimateRowUpperBound(distinct.Input, context, out upperBound);

                    case OffsetNode offset:
                        return TryEstimateRowUpperBound(offset.Input, context, out upperBound);

                    case UnionNode union:
                    {
                        long sum = 0;
                        foreach (var source in union.Sources)
                        {
                            if (!TryEstimateRowUpperBound(source, context, out var sourceUpperBound))
                            {
                                return false;
                            }

                            sum += sourceUpperBound;
                            if (sum > int.MaxValue)
                            {
                                upperBound = int.MaxValue;
                                return true;
                            }
                        }

                        upperBound = (int)sum;
                        return true;
                    }

                    case KeyJoinNode keyJoin:
                    {
                        if (TryEstimateRowUpperBound(keyJoin.Left, context, out var leftUpperBound) &&
                            TryEstimateRowUpperBound(keyJoin.Right, context, out var rightUpperBound))
                        {
                            var product = (long)leftUpperBound * rightUpperBound;
                            upperBound = product > int.MaxValue ? int.MaxValue : (int)product;
                            return true;
                        }

                        return false;
                    }

                    case JoinNode join:
                    {
                        if (TryEstimateRowUpperBound(join.Left, context, out var leftUpperBound) &&
                            TryEstimateRowUpperBound(join.Right, context, out var rightUpperBound))
                        {
                            var product = (long)leftUpperBound * rightUpperBound;
                            upperBound = product > int.MaxValue ? int.MaxValue : (int)product;
                            return true;
                        }

                        return false;
                    }

                    case RelationScanNode scan:
                        upperBound = GetFactsList(scan.Relation, context).Count;
                        return true;

                    case PatternScanNode scan:
                    {
                        var facts = GetFactsList(scan.Relation, context);
                        upperBound = facts.Count;

                        List<int>? boundColumns = null;
                        for (var i = 0; i < scan.Pattern.Length; i++)
                        {
                            var value = scan.Pattern[i];
                            if (ReferenceEquals(value, Wildcard.Value))
                            {
                                continue;
                            }

                            boundColumns ??= new List<int>();
                            boundColumns.Add(i);
                        }

                        if (boundColumns is null || boundColumns.Count == 0)
                        {
                            return true;
                        }

                        if (boundColumns.Count == 1)
                        {
                            var boundColumn = boundColumns[0];
                            var index = GetFactIndex(scan.Relation, boundColumn, facts, context);
                            var keyValue = scan.Pattern[boundColumn];
                            var lookupKey = keyValue ?? NullFactIndexKey;
                            upperBound = index.TryGetValue(lookupKey, out var bucket) ? bucket.Count : 0;

                            return true;
                        }

                        var signature = string.Join(",", boundColumns);
                        if (context.JoinIndices.TryGetValue((scan.Relation, signature), out var joinIndex))
                        {
                            var key = new object[boundColumns.Count];
                            for (var i = 0; i < boundColumns.Count; i++)
                            {
                                key[i] = scan.Pattern[boundColumns[i]];
                            }

                            var wrapper = new RowWrapper(key);
                            upperBound = joinIndex.TryGetValue(wrapper, out var bucket) ? bucket.Count : 0;
                            return true;
                        }

                        int? bestUpperBound = null;
                        foreach (var boundColumn in boundColumns)
                        {
                            var index = GetFactIndex(scan.Relation, boundColumn, facts, context);
                            var keyValue = scan.Pattern[boundColumn];
                            var lookupKey = keyValue ?? NullFactIndexKey;
                            var candidate = index.TryGetValue(lookupKey, out var bucket) ? bucket.Count : 0;
                            bestUpperBound = bestUpperBound is null ? candidate : Math.Min(bestUpperBound.Value, candidate);
                        }

                        if (bestUpperBound is not null)
                        {
                            upperBound = bestUpperBound.Value;
                        }

                        return true;
                    }

                    case RecursiveRefNode recursive:
                    {
                        var map = recursive.Kind == RecursiveRefKind.Total ? context.Totals : context.Deltas;
                        upperBound = map.TryGetValue(recursive.Predicate, out var rows) ? rows.Count : 0;
                        return true;
                    }

                    case CrossRefNode cross:
                    {
                        var map = cross.Kind == RecursiveRefKind.Total ? context.Totals : context.Deltas;
                        upperBound = map.TryGetValue(cross.Predicate, out var rows) ? rows.Count : 0;
                        return true;
                    }

                    case MaterializeNode materialize:
                    {
                        if (context.Materialized.TryGetValue(materialize.Id, out var cached))
                        {
                            upperBound = cached.Count;
                            return true;
                        }

                        return TryEstimateRowUpperBound(materialize.Plan, context, out upperBound);
                    }

                    default:
                        return false;
                }
            }

            static bool IsRecursiveProbe(PlanNode node) => node switch
            {
                RecursiveRefNode => true,
                CrossRefNode => true,
                MaterializeNode materialize => IsRecursiveProbe(materialize.Plan),
                ProgramNode program => program.Definitions.Any(IsRecursiveProbe) || IsRecursiveProbe(program.Body),
                DefineRelationNode define => IsRecursiveProbe(define.Plan),
                DefineMutualFixpointNode defineMutual => IsRecursiveProbe(defineMutual.Fixpoint),
                FixpointNode fixpoint => IsRecursiveProbe(fixpoint.BasePlan) || fixpoint.RecursivePlans.Any(IsRecursiveProbe),
                MutualFixpointNode mutual => mutual.Members.Any(member =>
                    IsRecursiveProbe(member.BasePlan) || member.RecursivePlans.Any(IsRecursiveProbe)),
                ProjectionNode projection => IsRecursiveProbe(projection.Input),
                SelectionNode selection => IsRecursiveProbe(selection.Input),
                ArithmeticNode arithmetic => IsRecursiveProbe(arithmetic.Input),
                NegationNode negation => IsRecursiveProbe(negation.Input),
                AggregateNode aggregate => IsRecursiveProbe(aggregate.Input),
                AggregateSubplanNode aggregateSubplan => IsRecursiveProbe(aggregateSubplan.Input) || IsRecursiveProbe(aggregateSubplan.Subplan),
                KeyJoinNode keyJoin => IsRecursiveProbe(keyJoin.Left) || IsRecursiveProbe(keyJoin.Right),
                JoinNode nestedJoin => IsRecursiveProbe(nestedJoin.Left) || IsRecursiveProbe(nestedJoin.Right),
                UnionNode union => union.Sources.Any(IsRecursiveProbe),
                DistinctNode distinct => IsRecursiveProbe(distinct.Input),
                OrderByNode orderBy => IsRecursiveProbe(orderBy.Input),
                LimitNode limit => IsRecursiveProbe(limit.Input),
                OffsetNode offset => IsRecursiveProbe(offset.Input),
                _ => false
            };

            if (context is not null)
            {
                var leftIsScan = TryGetPredicateScan(join.Left, out var leftScanPredicate, out var leftScanPattern, out var leftScanFilter);
                var rightIsScan = TryGetPredicateScan(join.Right, out var rightScanPredicate, out var rightScanPattern, out var rightScanFilter);

                if (leftIsScan || rightIsScan)
                {
                    var useScanIndexStrategy = true;
                    if (leftIsScan != rightIsScan)
                    {
                        var scanPredicate = leftIsScan ? leftScanPredicate : rightScanPredicate;
                        var scanPattern = leftIsScan ? leftScanPattern : rightScanPattern;
                        var scanKeys = leftIsScan ? join.LeftKeys : join.RightKeys;
                        var scanIndexKeys = GetScanIndexKeys(scanKeys, scanPattern);
                        var signature = scanIndexKeys.Count == 1 ? string.Empty : string.Join(",", scanIndexKeys);
                        var scanIndexCached = scanIndexKeys.Count == 1
                            ? context.FactIndices.ContainsKey((scanPredicate, scanIndexKeys[0]))
                            : context.JoinIndices.ContainsKey((scanPredicate, signature));

                        var otherNode = leftIsScan ? join.Right : join.Left;

                        if (context.FixpointDepth == 0 && !scanIndexCached && _cacheContext is null && !IsRecursiveProbe(otherNode))
                        {
                            const int TinyProbeUpperBound = 64;
                            if (joinKeyCount == 1 &&
                                     TryEstimateRowUpperBound(otherNode, context, out var probeUpperBound) &&
                                     probeUpperBound <= TinyProbeUpperBound)
                            {
                                useScanIndexStrategy = false;
                                forceBuildLeft = !leftIsScan;
                            }
                        }
                    }

                    if (useScanIndexStrategy && leftIsScan && rightIsScan)
                    {
                        var leftFacts = GetFactsList(leftScanPredicate, context);
                        var rightFacts = GetFactsList(rightScanPredicate, context);
                        var keyCount = join.LeftKeys.Count;
                        var leftIndexKeys = GetScanIndexKeys(join.LeftKeys, leftScanPattern);
                        var rightIndexKeys = GetScanIndexKeys(join.RightKeys, rightScanPattern);
                        var leftSignature = leftIndexKeys.Count == 1 ? string.Empty : string.Join(",", leftIndexKeys);
                        var rightSignature = rightIndexKeys.Count == 1 ? string.Empty : string.Join(",", rightIndexKeys);

                        static int CountMatchingPattern(IReadOnlyList<object[]> candidates, object[] pattern)
                        {
                            var count = 0;
                            foreach (var tuple in candidates)
                            {
                                if (tuple is null) continue;
                                if (TupleMatchesPattern(tuple, pattern))
                                {
                                    count++;
                                }
                            }

                            return count;
                        }

                        var estimatedLeftProbe = leftScanPattern is null
                            ? leftFacts.Count
                            : CountMatchingPattern(
                                SelectFactsForPattern(leftScanPredicate, leftFacts, leftScanPattern, context),
                                leftScanPattern);

                        var estimatedRightProbe = rightScanPattern is null
                            ? rightFacts.Count
                            : CountMatchingPattern(
                                SelectFactsForPattern(rightScanPredicate, rightFacts, rightScanPattern, context),
                                rightScanPattern);

                        var buildScanOnLeft = (long)leftFacts.Count + estimatedRightProbe <= (long)rightFacts.Count + estimatedLeftProbe;

                        var leftIndexCached = leftIndexKeys.Count == 1
                            ? context.FactIndices.ContainsKey((leftScanPredicate, leftIndexKeys[0]))
                            : context.JoinIndices.ContainsKey((leftScanPredicate, leftSignature));

                        var rightIndexCached = rightIndexKeys.Count == 1
                            ? context.FactIndices.ContainsKey((rightScanPredicate, rightIndexKeys[0]))
                            : context.JoinIndices.ContainsKey((rightScanPredicate, rightSignature));

                        if (leftIndexCached != rightIndexCached)
                        {
                            buildScanOnLeft = leftIndexCached;
                        }

                        if (buildScanOnLeft)
                        {
                            if (leftIndexKeys.Count == 1)
                            {
                                trace?.RecordStrategy(join, "KeyJoinScanIndex");
                                var probe = Evaluate(join.Right, context);
                                var index = GetFactIndex(leftScanPredicate, join.LeftKeys[0], leftFacts, context);
                                foreach (var rightTuple in probe)
                                {
                                    if (rightTuple is null) continue;
                                    if (!ProbeMatchesScanJoinKeyConstants(rightTuple, join.RightKeys, join.LeftKeys, leftScanPattern)) continue;

                                    var lookupKey = GetLookupKey(rightTuple, join.RightKeys[0], NullFactIndexKey);
                                    if (!index.TryGetValue(lookupKey, out var bucket))
                                    {
                                        continue;
                                    }

                                    foreach (var leftTuple in bucket)
                                    {
                                        if (leftTuple is null) continue;
                                        if (leftScanFilter is not null && !leftScanFilter(leftTuple)) continue;
                                        yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                    }
                                }
                            }
                            else
                            {
                                var joinIndexCached = context.JoinIndices.ContainsKey((leftScanPredicate, leftSignature));

                                const int TinyProbeUpperBound = 64;
                                if (!joinIndexCached &&
                                    keyCount > 1 &&
                                    (context.FixpointDepth == 0 || IsRecursiveProbe(join.Right)))
                                {
                                    var probeSource = Evaluate(join.Right, context);
                                    using var probeEnumerator = probeSource.GetEnumerator();
                                    var probeRows = new List<(object[] Tuple, object[] Keys)>();
                                    var distinctKeys = new HashSet<object>[joinKeyCount];
                                    for (var i = 0; i < joinKeyCount; i++)
                                    {
                                        distinctKeys[i] = new HashSet<object>();
                                    }

                                    var probeTooLarge = false;
                                    while (probeEnumerator.MoveNext())
                                    {
                                        var rightTuple = probeEnumerator.Current;
                                        if (rightTuple is null) continue;
                                        if (!ProbeMatchesScanJoinKeyConstants(rightTuple, join.RightKeys, join.LeftKeys, leftScanPattern)) continue;

                                        var keyValues = new object[joinKeyCount];
                                        for (var i = 0; i < joinKeyCount; i++)
                                        {
                                            var key = GetLookupKey(rightTuple, join.RightKeys[i], NullFactIndexKey);
                                            keyValues[i] = key;
                                            distinctKeys[i].Add(key);
                                        }

                                        probeRows.Add((rightTuple, keyValues));
                                        if (probeRows.Count > TinyProbeUpperBound)
                                        {
                                            probeTooLarge = true;
                                            break;
                                        }
                                    }

                                    if (!probeTooLarge)
                                    {
                                        trace?.RecordStrategy(join, "KeyJoinScanIndexPartial");
                                        if (probeRows.Count == 0)
                                        {
                                            yield break;
                                        }

                                        var pivotPos = 0;
                                        var bestDistinct = distinctKeys[0].Count;
                                        var bestCached = context.FactIndices.ContainsKey((leftScanPredicate, join.LeftKeys[0]));
                                        for (var i = 1; i < joinKeyCount; i++)
                                        {
                                            var candidateDistinct = distinctKeys[i].Count;
                                            var candidateCached = context.FactIndices.ContainsKey((leftScanPredicate, join.LeftKeys[i]));

                                            if (candidateCached && !bestCached)
                                            {
                                                pivotPos = i;
                                                bestDistinct = candidateDistinct;
                                                bestCached = true;
                                                continue;
                                            }

                                            if (candidateCached == bestCached && candidateDistinct > bestDistinct)
                                            {
                                                pivotPos = i;
                                                bestDistinct = candidateDistinct;
                                            }
                                        }

                                        var pivotLeftKey = join.LeftKeys[pivotPos];

                                        var probeIndex = new Dictionary<object, List<(object[] Tuple, object[] Keys)>>();
                                        foreach (var probeRow in probeRows)
                                        {
                                            var pivotKey = probeRow.Keys[pivotPos];
                                            if (!probeIndex.TryGetValue(pivotKey, out var bucket))
                                            {
                                                bucket = new List<(object[] Tuple, object[] Keys)>();
                                                probeIndex[pivotKey] = bucket;
                                            }

                                            bucket.Add(probeRow);
                                        }

                                        if (probeIndex.Count == 0)
                                        {
                                            yield break;
                                        }

                                        var factIndex = GetFactIndex(leftScanPredicate, pivotLeftKey, leftFacts, context);
                                        foreach (var entry in probeIndex)
                                        {
                                            if (!factIndex.TryGetValue(entry.Key, out var leftBucket))
                                            {
                                                continue;
                                            }

                                            foreach (var leftTuple in leftBucket)
                                            {
                                                if (leftTuple is null) continue;
                                                if (leftScanFilter is not null && !leftScanFilter(leftTuple)) continue;
                                                if (leftScanPattern is not null && !TupleMatchesPattern(leftTuple, leftScanPattern)) continue;

                                                foreach (var probeRow in entry.Value)
                                                {
                                                    var match = true;
                                                    for (var i = 0; i < joinKeyCount; i++)
                                                    {
                                                        if (i == pivotPos) continue;
                                                        var leftValue = GetLookupKey(leftTuple, join.LeftKeys[i], NullFactIndexKey);
                                                        if (!Equals(leftValue, probeRow.Keys[i]))
                                                        {
                                                            match = false;
                                                            break;
                                                        }
                                                    }

                                                    if (!match)
                                                    {
                                                        continue;
                                                    }

                                                    yield return BuildJoinOutput(leftTuple, probeRow.Tuple, join.LeftWidth, join.RightWidth, join.Width);
                                                }
                                            }
                                        }

                                        yield break;
                                    }

                                    trace?.RecordStrategy(join, "KeyJoinScanIndex");
                                    var scanScanIndexFallbackLeft = GetJoinIndex(leftScanPredicate, leftIndexKeys, leftFacts, context);

                                    foreach (var probeRow in probeRows)
                                    {
                                        var rightTuple = probeRow.Tuple;
                                        if (rightTuple is null) continue;

                                        var key = leftIndexKeys.Count == keyCount
                                            ? BuildKeyFromTuple(rightTuple, join.RightKeys)
                                            : BuildScanJoinLookupKey(rightTuple, join.RightKeys, join.LeftKeys, leftIndexKeys, leftScanPattern);
                                        var wrapper = new RowWrapper(key);

                                        if (!scanScanIndexFallbackLeft.TryGetValue(wrapper, out var bucket))
                                        {
                                            continue;
                                        }

                                        foreach (var leftTuple in bucket)
                                        {
                                            if (leftTuple is null) continue;
                                            if (leftScanFilter is not null && !leftScanFilter(leftTuple)) continue;
                                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                        }
                                    }

                                    while (probeEnumerator.MoveNext())
                                    {
                                        var rightTuple = probeEnumerator.Current;
                                        if (rightTuple is null) continue;
                                        if (!ProbeMatchesScanJoinKeyConstants(rightTuple, join.RightKeys, join.LeftKeys, leftScanPattern)) continue;

                                        var key = leftIndexKeys.Count == keyCount
                                            ? BuildKeyFromTuple(rightTuple, join.RightKeys)
                                            : BuildScanJoinLookupKey(rightTuple, join.RightKeys, join.LeftKeys, leftIndexKeys, leftScanPattern);
                                        var wrapper = new RowWrapper(key);

                                        if (!scanScanIndexFallbackLeft.TryGetValue(wrapper, out var bucket))
                                        {
                                            continue;
                                        }

                                        foreach (var leftTuple in bucket)
                                        {
                                            if (leftTuple is null) continue;
                                            if (leftScanFilter is not null && !leftScanFilter(leftTuple)) continue;
                                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                        }
                                    }

                                    yield break;
                                }

                                trace?.RecordStrategy(join, "KeyJoinScanIndex");
                                var index = GetJoinIndex(leftScanPredicate, leftIndexKeys, leftFacts, context);
                                var probe = Evaluate(join.Right, context);
                                foreach (var rightTuple in probe)
                                {
                                    if (rightTuple is null) continue;
                                    if (!ProbeMatchesScanJoinKeyConstants(rightTuple, join.RightKeys, join.LeftKeys, leftScanPattern)) continue;

                                    var key = leftIndexKeys.Count == keyCount
                                        ? BuildKeyFromTuple(rightTuple, join.RightKeys)
                                        : BuildScanJoinLookupKey(rightTuple, join.RightKeys, join.LeftKeys, leftIndexKeys, leftScanPattern);
                                    var wrapper = new RowWrapper(key);

                                    if (!index.TryGetValue(wrapper, out var bucket))
                                    {
                                        continue;
                                    }

                                    foreach (var leftTuple in bucket)
                                    {
                                        if (leftTuple is null) continue;
                                        if (leftScanFilter is not null && !leftScanFilter(leftTuple)) continue;
                                        yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                    }
                                }
                            }
                        }
                        else
                        {
                            if (rightIndexKeys.Count == 1)
                            {
                                trace?.RecordStrategy(join, "KeyJoinScanIndex");
                                var probe = Evaluate(join.Left, context);
                                var index = GetFactIndex(rightScanPredicate, join.RightKeys[0], rightFacts, context);
                                foreach (var leftTuple in probe)
                                {
                                    if (leftTuple is null) continue;
                                    if (!ProbeMatchesScanJoinKeyConstants(leftTuple, join.LeftKeys, join.RightKeys, rightScanPattern)) continue;

                                    var lookupKey = GetLookupKey(leftTuple, join.LeftKeys[0], NullFactIndexKey);
                                    if (!index.TryGetValue(lookupKey, out var bucket))
                                    {
                                        continue;
                                    }

                                    foreach (var rightTuple in bucket)
                                    {
                                        if (rightTuple is null) continue;
                                        if (rightScanFilter is not null && !rightScanFilter(rightTuple)) continue;
                                        yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                    }
                                }
                            }
                            else
                            {
                                var joinIndexCached = context.JoinIndices.ContainsKey((rightScanPredicate, rightSignature));

                                const int TinyProbeUpperBound = 64;
                                if (!joinIndexCached &&
                                    keyCount > 1 &&
                                    (context.FixpointDepth == 0 || IsRecursiveProbe(join.Left)))
                                {
                                    var probeSource = Evaluate(join.Left, context);
                                    using var probeEnumerator = probeSource.GetEnumerator();
                                    var probeRows = new List<(object[] Tuple, object[] Keys)>();
                                    var distinctKeys = new HashSet<object>[joinKeyCount];
                                    for (var i = 0; i < joinKeyCount; i++)
                                    {
                                        distinctKeys[i] = new HashSet<object>();
                                    }

                                    var probeTooLarge = false;
                                    while (probeEnumerator.MoveNext())
                                    {
                                        var leftTuple = probeEnumerator.Current;
                                        if (leftTuple is null) continue;
                                        if (!ProbeMatchesScanJoinKeyConstants(leftTuple, join.LeftKeys, join.RightKeys, rightScanPattern)) continue;

                                        var keyValues = new object[joinKeyCount];
                                        for (var i = 0; i < joinKeyCount; i++)
                                        {
                                            var key = GetLookupKey(leftTuple, join.LeftKeys[i], NullFactIndexKey);
                                            keyValues[i] = key;
                                            distinctKeys[i].Add(key);
                                        }

                                        probeRows.Add((leftTuple, keyValues));
                                        if (probeRows.Count > TinyProbeUpperBound)
                                        {
                                            probeTooLarge = true;
                                            break;
                                        }
                                    }

                                    if (!probeTooLarge)
                                    {
                                        trace?.RecordStrategy(join, "KeyJoinScanIndexPartial");
                                        if (probeRows.Count == 0)
                                        {
                                            yield break;
                                        }

                                        var pivotPos = 0;
                                        var bestDistinct = distinctKeys[0].Count;
                                        var bestCached = context.FactIndices.ContainsKey((rightScanPredicate, join.RightKeys[0]));
                                        for (var i = 1; i < joinKeyCount; i++)
                                        {
                                            var candidateDistinct = distinctKeys[i].Count;
                                            var candidateCached = context.FactIndices.ContainsKey((rightScanPredicate, join.RightKeys[i]));

                                            if (candidateCached && !bestCached)
                                            {
                                                pivotPos = i;
                                                bestDistinct = candidateDistinct;
                                                bestCached = true;
                                                continue;
                                            }

                                            if (candidateCached == bestCached && candidateDistinct > bestDistinct)
                                            {
                                                pivotPos = i;
                                                bestDistinct = candidateDistinct;
                                            }
                                        }

                                        var pivotRightKey = join.RightKeys[pivotPos];

                                        var probeIndex = new Dictionary<object, List<(object[] Tuple, object[] Keys)>>();
                                        foreach (var probeRow in probeRows)
                                        {
                                            var pivotKey = probeRow.Keys[pivotPos];
                                            if (!probeIndex.TryGetValue(pivotKey, out var bucket))
                                            {
                                                bucket = new List<(object[] Tuple, object[] Keys)>();
                                                probeIndex[pivotKey] = bucket;
                                            }

                                            bucket.Add(probeRow);
                                        }

                                        if (probeIndex.Count == 0)
                                        {
                                            yield break;
                                        }

                                        var factIndex = GetFactIndex(rightScanPredicate, pivotRightKey, rightFacts, context);
                                        foreach (var entry in probeIndex)
                                        {
                                            if (!factIndex.TryGetValue(entry.Key, out var rightBucket))
                                            {
                                                continue;
                                            }

                                            foreach (var rightTuple in rightBucket)
                                            {
                                                if (rightTuple is null) continue;
                                                if (rightScanFilter is not null && !rightScanFilter(rightTuple)) continue;
                                                if (rightScanPattern is not null && !TupleMatchesPattern(rightTuple, rightScanPattern)) continue;

                                                foreach (var probeRow in entry.Value)
                                                {
                                                    var match = true;
                                                    for (var i = 0; i < joinKeyCount; i++)
                                                    {
                                                        if (i == pivotPos) continue;
                                                        var rightValue = GetLookupKey(rightTuple, join.RightKeys[i], NullFactIndexKey);
                                                        if (!Equals(rightValue, probeRow.Keys[i]))
                                                        {
                                                            match = false;
                                                            break;
                                                        }
                                                    }

                                                    if (!match)
                                                    {
                                                        continue;
                                                    }

                                                    yield return BuildJoinOutput(probeRow.Tuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                                }
                                            }
                                        }

                                        yield break;
                                    }

                                    trace?.RecordStrategy(join, "KeyJoinScanIndex");
                                    var scanScanIndexFallbackRight = GetJoinIndex(rightScanPredicate, rightIndexKeys, rightFacts, context);

                                    foreach (var probeRow in probeRows)
                                    {
                                        var leftTuple = probeRow.Tuple;
                                        if (leftTuple is null) continue;

                                        var key = rightIndexKeys.Count == keyCount
                                            ? BuildKeyFromTuple(leftTuple, join.LeftKeys)
                                            : BuildScanJoinLookupKey(leftTuple, join.LeftKeys, join.RightKeys, rightIndexKeys, rightScanPattern);
                                        var wrapper = new RowWrapper(key);

                                        if (!scanScanIndexFallbackRight.TryGetValue(wrapper, out var bucket))
                                        {
                                            continue;
                                        }

                                        foreach (var rightTuple in bucket)
                                        {
                                            if (rightTuple is null) continue;
                                            if (rightScanFilter is not null && !rightScanFilter(rightTuple)) continue;
                                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                        }
                                    }

                                    while (probeEnumerator.MoveNext())
                                    {
                                        var leftTuple = probeEnumerator.Current;
                                        if (leftTuple is null) continue;
                                        if (!ProbeMatchesScanJoinKeyConstants(leftTuple, join.LeftKeys, join.RightKeys, rightScanPattern)) continue;

                                        var key = rightIndexKeys.Count == keyCount
                                            ? BuildKeyFromTuple(leftTuple, join.LeftKeys)
                                            : BuildScanJoinLookupKey(leftTuple, join.LeftKeys, join.RightKeys, rightIndexKeys, rightScanPattern);
                                        var wrapper = new RowWrapper(key);

                                        if (!scanScanIndexFallbackRight.TryGetValue(wrapper, out var bucket))
                                        {
                                            continue;
                                        }

                                        foreach (var rightTuple in bucket)
                                        {
                                            if (rightTuple is null) continue;
                                            if (rightScanFilter is not null && !rightScanFilter(rightTuple)) continue;
                                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                        }
                                    }

                                    yield break;
                                }

                                trace?.RecordStrategy(join, "KeyJoinScanIndex");
                                var index = GetJoinIndex(rightScanPredicate, rightIndexKeys, rightFacts, context);
                                var probe = Evaluate(join.Left, context);
                                foreach (var leftTuple in probe)
                                {
                                    if (leftTuple is null) continue;
                                    if (!ProbeMatchesScanJoinKeyConstants(leftTuple, join.LeftKeys, join.RightKeys, rightScanPattern)) continue;

                                    var key = rightIndexKeys.Count == keyCount
                                        ? BuildKeyFromTuple(leftTuple, join.LeftKeys)
                                        : BuildScanJoinLookupKey(leftTuple, join.LeftKeys, join.RightKeys, rightIndexKeys, rightScanPattern);
                                    var wrapper = new RowWrapper(key);

                                    if (!index.TryGetValue(wrapper, out var bucket))
                                    {
                                        continue;
                                    }

                                    foreach (var rightTuple in bucket)
                                    {
                                        if (rightTuple is null) continue;
                                        if (rightScanFilter is not null && !rightScanFilter(rightTuple)) continue;
                                        yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                    }
                                }
                            }
                        }

                        yield break;
                    }

                    if (useScanIndexStrategy && rightIsScan)
                    {
                        var facts = GetFactsList(rightScanPredicate, context);
                        var keyCount = join.RightKeys.Count;
                        var rightIndexKeys = GetScanIndexKeys(join.RightKeys, rightScanPattern);

                        if (rightIndexKeys.Count == 1)
                        {
                            const int TinyProbeUpperBound = 64;
                            var scanIndexCached = context.FactIndices.ContainsKey((rightScanPredicate, rightIndexKeys[0]));
                            var allowProbeSampling = context.FixpointDepth == 0 && !scanIndexCached && _cacheContext is null && !IsRecursiveProbe(join.Left);

                            if (allowProbeSampling && joinKeyCount == 1)
                            {
                                var probeSource = Evaluate(join.Left, context);
                                using var probeEnumerator = probeSource.GetEnumerator();
                                var probeRows = new List<object[]>();

                                var probeTooLarge = false;
                                while (probeEnumerator.MoveNext())
                                {
                                    var leftTuple = probeEnumerator.Current;
                                    if (leftTuple is null) continue;
                                    if (!ProbeMatchesScanJoinKeyConstants(leftTuple, join.LeftKeys, join.RightKeys, rightScanPattern)) continue;

                                    probeRows.Add(leftTuple);
                                    if (probeRows.Count > TinyProbeUpperBound)
                                    {
                                        probeTooLarge = true;
                                        break;
                                    }
                                }

                                if (!probeTooLarge)
                                {
                                    trace?.RecordStrategy(join, "KeyJoinHashBuildLeft");

                                    var probeIndex = new Dictionary<object, List<object[]>>();
                                    foreach (var leftTuple in probeRows)
                                    {
                                        var key = GetLookupKey(leftTuple, join.LeftKeys[0], NullFactIndexKey);
                                        if (!probeIndex.TryGetValue(key, out var bucket))
                                        {
                                            bucket = new List<object[]>();
                                            probeIndex[key] = bucket;
                                        }

                                        bucket.Add(leftTuple);
                                    }

                                    if (probeIndex.Count == 0)
                                    {
                                        yield break;
                                    }

                                    foreach (var rightTuple in facts)
                                    {
                                        if (rightTuple is null) continue;
                                        if (rightScanFilter is not null && !rightScanFilter(rightTuple)) continue;
                                        if (rightScanPattern is not null && !TupleMatchesPattern(rightTuple, rightScanPattern)) continue;

                                        var key = GetLookupKey(rightTuple, join.RightKeys[0], NullFactIndexKey);
                                        if (!probeIndex.TryGetValue(key, out var bucket))
                                        {
                                            continue;
                                        }

                                        foreach (var leftTuple in bucket)
                                        {
                                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                        }
                                    }

                                    yield break;
                                }

                                trace?.RecordStrategy(join, "KeyJoinScanIndex");
                                var index = GetFactIndex(rightScanPredicate, join.RightKeys[0], facts, context);

                                foreach (var leftTuple in probeRows)
                                {
                                    if (leftTuple is null) continue;

                                    var lookupKey = GetLookupKey(leftTuple, join.LeftKeys[0], NullFactIndexKey);
                                    if (!index.TryGetValue(lookupKey, out var bucket))
                                    {
                                        continue;
                                    }

                                    foreach (var rightTuple in bucket)
                                    {
                                        if (rightTuple is null) continue;
                                        if (rightScanFilter is not null && !rightScanFilter(rightTuple)) continue;
                                        yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                    }
                                }

                                while (probeEnumerator.MoveNext())
                                {
                                    var leftTuple = probeEnumerator.Current;
                                    if (leftTuple is null) continue;
                                    if (!ProbeMatchesScanJoinKeyConstants(leftTuple, join.LeftKeys, join.RightKeys, rightScanPattern)) continue;

                                    var lookupKey = GetLookupKey(leftTuple, join.LeftKeys[0], NullFactIndexKey);
                                    if (!index.TryGetValue(lookupKey, out var bucket))
                                    {
                                        continue;
                                    }

                                    foreach (var rightTuple in bucket)
                                    {
                                        if (rightTuple is null) continue;
                                        if (rightScanFilter is not null && !rightScanFilter(rightTuple)) continue;
                                        yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                    }
                                }

                                yield break;
                            }

                            trace?.RecordStrategy(join, "KeyJoinScanIndex");
                            var probe = Evaluate(join.Left, context);
                            var fallbackIndex = GetFactIndex(rightScanPredicate, join.RightKeys[0], facts, context);
                            foreach (var leftTuple in probe)
                            {
                                if (leftTuple is null) continue;
                                if (!ProbeMatchesScanJoinKeyConstants(leftTuple, join.LeftKeys, join.RightKeys, rightScanPattern)) continue;

                                var lookupKey = GetLookupKey(leftTuple, join.LeftKeys[0], NullFactIndexKey);
                                if (!fallbackIndex.TryGetValue(lookupKey, out var bucket))
                                {
                                    continue;
                                }

                                foreach (var rightTuple in bucket)
                                {
                                    if (rightTuple is null) continue;
                                    if (rightScanFilter is not null && !rightScanFilter(rightTuple)) continue;
                                    yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                }
                            }
                        }
                        else
                        {
                            var signature = string.Join(",", rightIndexKeys);
                            var joinIndexCached = context.JoinIndices.ContainsKey((rightScanPredicate, signature));

                            const int TinyProbeUpperBound = 64;
                            if (!joinIndexCached &&
                                keyCount > 1 &&
                                (context.FixpointDepth == 0 || IsRecursiveProbe(join.Left)))
                            {
                                var probeSource = Evaluate(join.Left, context);
                                using var probeEnumerator = probeSource.GetEnumerator();
                                var probeRows = new List<(object[] Tuple, object[] Keys)>();
                                var distinctKeys = new HashSet<object>[joinKeyCount];
                                for (var i = 0; i < joinKeyCount; i++)
                                {
                                    distinctKeys[i] = new HashSet<object>();
                                }

                                var probeTooLarge = false;
                                while (probeEnumerator.MoveNext())
                                {
                                    var leftTuple = probeEnumerator.Current;
                                    if (leftTuple is null) continue;
                                    if (!ProbeMatchesScanJoinKeyConstants(leftTuple, join.LeftKeys, join.RightKeys, rightScanPattern)) continue;

                                    var keyValues = new object[joinKeyCount];
                                    for (var i = 0; i < joinKeyCount; i++)
                                    {
                                        var key = GetLookupKey(leftTuple, join.LeftKeys[i], NullFactIndexKey);
                                        keyValues[i] = key;
                                        distinctKeys[i].Add(key);
                                    }

                                    probeRows.Add((leftTuple, keyValues));
                                    if (probeRows.Count > TinyProbeUpperBound)
                                    {
                                        probeTooLarge = true;
                                        break;
                                    }
                                }

                                if (!probeTooLarge)
                                {
                                    trace?.RecordStrategy(join, "KeyJoinScanIndexPartial");
                                    if (probeRows.Count == 0)
                                    {
                                        yield break;
                                    }

                                    var pivotPos = 0;
                                    var bestDistinct = distinctKeys[0].Count;
                                    var bestCached = context.FactIndices.ContainsKey((rightScanPredicate, join.RightKeys[0]));
                                    for (var i = 1; i < joinKeyCount; i++)
                                    {
                                        var candidateDistinct = distinctKeys[i].Count;
                                        var candidateCached = context.FactIndices.ContainsKey((rightScanPredicate, join.RightKeys[i]));

                                        if (candidateCached && !bestCached)
                                        {
                                            pivotPos = i;
                                            bestDistinct = candidateDistinct;
                                            bestCached = true;
                                            continue;
                                        }

                                        if (candidateCached == bestCached && candidateDistinct > bestDistinct)
                                        {
                                            pivotPos = i;
                                            bestDistinct = candidateDistinct;
                                        }
                                    }

                                    var pivotRightKey = join.RightKeys[pivotPos];

                                var probeIndex = new Dictionary<object, List<(object[] Tuple, object[] Keys)>>();
                                foreach (var probeRow in probeRows)
                                {
                                    var pivotKey = probeRow.Keys[pivotPos];
                                    if (!probeIndex.TryGetValue(pivotKey, out var bucket))
                                    {
                                        bucket = new List<(object[] Tuple, object[] Keys)>();
                                        probeIndex[pivotKey] = bucket;
                                    }

                                    bucket.Add(probeRow);
                                }

                                if (probeIndex.Count == 0)
                                {
                                    yield break;
                                }

                                var factIndex = GetFactIndex(rightScanPredicate, pivotRightKey, facts, context);
                                foreach (var entry in probeIndex)
                                {
                                    if (!factIndex.TryGetValue(entry.Key, out var rightBucket))
                                    {
                                        continue;
                                    }

                                    foreach (var rightTuple in rightBucket)
                                    {
                                        if (rightTuple is null) continue;
                                        if (rightScanFilter is not null && !rightScanFilter(rightTuple)) continue;
                                        if (rightScanPattern is not null && !TupleMatchesPattern(rightTuple, rightScanPattern)) continue;

                                        foreach (var probeRow in entry.Value)
                                        {
                                            var match = true;
                                            for (var i = 0; i < joinKeyCount; i++)
                                            {
                                                if (i == pivotPos) continue;
                                                var rightValue = GetLookupKey(rightTuple, join.RightKeys[i], NullFactIndexKey);
                                                if (!Equals(rightValue, probeRow.Keys[i]))
                                                {
                                                    match = false;
                                                    break;
                                                }
                                            }

                                            if (!match)
                                            {
                                                continue;
                                            }

                                            yield return BuildJoinOutput(probeRow.Tuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                        }
                                    }
                                }

                                    yield break;
                                }

                                trace?.RecordStrategy(join, "KeyJoinScanIndex");
                                var fallbackIndex = GetJoinIndex(rightScanPredicate, rightIndexKeys, facts, context);
                                foreach (var probeRow in probeRows)
                                {
                                    var leftTuple = probeRow.Tuple;
                                    if (leftTuple is null) continue;

                                    var key = rightIndexKeys.Count == keyCount
                                        ? BuildKeyFromTuple(leftTuple, join.LeftKeys)
                                        : BuildScanJoinLookupKey(leftTuple, join.LeftKeys, join.RightKeys, rightIndexKeys, rightScanPattern);
                                    var wrapper = new RowWrapper(key);

                                    if (!fallbackIndex.TryGetValue(wrapper, out var bucket))
                                    {
                                        continue;
                                    }

                                    foreach (var rightTuple in bucket)
                                    {
                                        if (rightTuple is null) continue;
                                        if (rightScanFilter is not null && !rightScanFilter(rightTuple)) continue;
                                        yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                    }
                                }

                                while (probeEnumerator.MoveNext())
                                {
                                    var leftTuple = probeEnumerator.Current;
                                    if (leftTuple is null) continue;
                                    if (!ProbeMatchesScanJoinKeyConstants(leftTuple, join.LeftKeys, join.RightKeys, rightScanPattern)) continue;

                                    var key = rightIndexKeys.Count == keyCount
                                        ? BuildKeyFromTuple(leftTuple, join.LeftKeys)
                                        : BuildScanJoinLookupKey(leftTuple, join.LeftKeys, join.RightKeys, rightIndexKeys, rightScanPattern);
                                    var wrapper = new RowWrapper(key);

                                    if (!fallbackIndex.TryGetValue(wrapper, out var bucket))
                                    {
                                        continue;
                                    }

                                    foreach (var rightTuple in bucket)
                                    {
                                        if (rightTuple is null) continue;
                                        if (rightScanFilter is not null && !rightScanFilter(rightTuple)) continue;
                                        yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                    }
                                }

                                yield break;
                            }

                            trace?.RecordStrategy(join, "KeyJoinScanIndex");
                            var index = GetJoinIndex(rightScanPredicate, rightIndexKeys, facts, context);
                            var probe = Evaluate(join.Left, context);
                            foreach (var leftTuple in probe)
                            {
                                if (leftTuple is null) continue;
                                if (!ProbeMatchesScanJoinKeyConstants(leftTuple, join.LeftKeys, join.RightKeys, rightScanPattern)) continue;

                                var key = rightIndexKeys.Count == keyCount
                                    ? BuildKeyFromTuple(leftTuple, join.LeftKeys)
                                    : BuildScanJoinLookupKey(leftTuple, join.LeftKeys, join.RightKeys, rightIndexKeys, rightScanPattern);
                                var wrapper = new RowWrapper(key);

                                if (!index.TryGetValue(wrapper, out var bucket))
                                {
                                    continue;
                                }

                                 foreach (var rightTuple in bucket)
                                 {
                                     if (rightTuple is null) continue;
                                     if (rightScanFilter is not null && !rightScanFilter(rightTuple)) continue;
                                     yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                 }
                            }
                        }

                        yield break;
                    }

                    if (useScanIndexStrategy && leftIsScan)
                    {
                        var facts = GetFactsList(leftScanPredicate, context);
                        var keyCount = join.LeftKeys.Count;
                        var leftIndexKeys = GetScanIndexKeys(join.LeftKeys, leftScanPattern);

                        if (leftIndexKeys.Count == 1)
                        {
                            const int TinyProbeUpperBound = 64;
                            var scanIndexCached = context.FactIndices.ContainsKey((leftScanPredicate, leftIndexKeys[0]));
                            var allowProbeSampling = context.FixpointDepth == 0 && !scanIndexCached && _cacheContext is null && !IsRecursiveProbe(join.Right);

                            if (allowProbeSampling && joinKeyCount == 1)
                            {
                                var probeSource = Evaluate(join.Right, context);
                                using var probeEnumerator = probeSource.GetEnumerator();
                                var probeRows = new List<object[]>();

                                var probeTooLarge = false;
                                while (probeEnumerator.MoveNext())
                                {
                                    var rightTuple = probeEnumerator.Current;
                                    if (rightTuple is null) continue;
                                    if (!ProbeMatchesScanJoinKeyConstants(rightTuple, join.RightKeys, join.LeftKeys, leftScanPattern)) continue;

                                    probeRows.Add(rightTuple);
                                    if (probeRows.Count > TinyProbeUpperBound)
                                    {
                                        probeTooLarge = true;
                                        break;
                                    }
                                }

                                if (!probeTooLarge)
                                {
                                    trace?.RecordStrategy(join, "KeyJoinHashBuildRight");

                                    var probeIndex = new Dictionary<object, List<object[]>>();
                                    foreach (var rightTuple in probeRows)
                                    {
                                        var key = GetLookupKey(rightTuple, join.RightKeys[0], NullFactIndexKey);
                                        if (!probeIndex.TryGetValue(key, out var bucket))
                                        {
                                            bucket = new List<object[]>();
                                            probeIndex[key] = bucket;
                                        }

                                        bucket.Add(rightTuple);
                                    }

                                    if (probeIndex.Count == 0)
                                    {
                                        yield break;
                                    }

                                    foreach (var leftTuple in facts)
                                    {
                                        if (leftTuple is null) continue;
                                        if (leftScanFilter is not null && !leftScanFilter(leftTuple)) continue;
                                        if (leftScanPattern is not null && !TupleMatchesPattern(leftTuple, leftScanPattern)) continue;

                                        var key = GetLookupKey(leftTuple, join.LeftKeys[0], NullFactIndexKey);
                                        if (!probeIndex.TryGetValue(key, out var bucket))
                                        {
                                            continue;
                                        }

                                        foreach (var rightTuple in bucket)
                                        {
                                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                        }
                                    }

                                    yield break;
                                }

                                trace?.RecordStrategy(join, "KeyJoinScanIndex");
                                var index = GetFactIndex(leftScanPredicate, join.LeftKeys[0], facts, context);

                                foreach (var rightTuple in probeRows)
                                {
                                    if (rightTuple is null) continue;

                                    var lookupKey = GetLookupKey(rightTuple, join.RightKeys[0], NullFactIndexKey);
                                    if (!index.TryGetValue(lookupKey, out var bucket))
                                    {
                                        continue;
                                    }

                                    foreach (var leftTuple in bucket)
                                    {
                                        if (leftTuple is null) continue;
                                        if (leftScanFilter is not null && !leftScanFilter(leftTuple)) continue;
                                        yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                    }
                                }

                                while (probeEnumerator.MoveNext())
                                {
                                    var rightTuple = probeEnumerator.Current;
                                    if (rightTuple is null) continue;
                                    if (!ProbeMatchesScanJoinKeyConstants(rightTuple, join.RightKeys, join.LeftKeys, leftScanPattern)) continue;

                                    var lookupKey = GetLookupKey(rightTuple, join.RightKeys[0], NullFactIndexKey);
                                    if (!index.TryGetValue(lookupKey, out var bucket))
                                    {
                                        continue;
                                    }

                                    foreach (var leftTuple in bucket)
                                    {
                                        if (leftTuple is null) continue;
                                        if (leftScanFilter is not null && !leftScanFilter(leftTuple)) continue;
                                        yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                    }
                                }

                                yield break;
                            }

                            trace?.RecordStrategy(join, "KeyJoinScanIndex");
                            var probe = Evaluate(join.Right, context);
                            var fallbackIndex = GetFactIndex(leftScanPredicate, join.LeftKeys[0], facts, context);
                            foreach (var rightTuple in probe)
                            {
                                if (rightTuple is null) continue;
                                if (!ProbeMatchesScanJoinKeyConstants(rightTuple, join.RightKeys, join.LeftKeys, leftScanPattern)) continue;

                                var lookupKey = GetLookupKey(rightTuple, join.RightKeys[0], NullFactIndexKey);
                                if (!fallbackIndex.TryGetValue(lookupKey, out var bucket))
                                {
                                    continue;
                                }

                                foreach (var leftTuple in bucket)
                                {
                                    if (leftTuple is null) continue;
                                    if (leftScanFilter is not null && !leftScanFilter(leftTuple)) continue;
                                    yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                }
                            }
                        }
                         else
                         {
                             var signature = string.Join(",", leftIndexKeys);
                             var joinIndexCached = context.JoinIndices.ContainsKey((leftScanPredicate, signature));

                             const int TinyProbeUpperBound = 64;
                             if (!joinIndexCached &&
                                 keyCount > 1 &&
                                 (context.FixpointDepth == 0 || IsRecursiveProbe(join.Right)))
                             {
                                 var probeSource = Evaluate(join.Right, context);
                                 using var probeEnumerator = probeSource.GetEnumerator();
                                 var probeRows = new List<(object[] Tuple, object[] Keys)>();
                                 var distinctKeys = new HashSet<object>[joinKeyCount];
                                 for (var i = 0; i < joinKeyCount; i++)
                                 {
                                     distinctKeys[i] = new HashSet<object>();
                                 }

                                 var probeTooLarge = false;
                                 while (probeEnumerator.MoveNext())
                                 {
                                     var rightTuple = probeEnumerator.Current;
                                     if (rightTuple is null) continue;
                                     if (!ProbeMatchesScanJoinKeyConstants(rightTuple, join.RightKeys, join.LeftKeys, leftScanPattern)) continue;

                                     var keyValues = new object[joinKeyCount];
                                     for (var i = 0; i < joinKeyCount; i++)
                                     {
                                         var key = GetLookupKey(rightTuple, join.RightKeys[i], NullFactIndexKey);
                                         keyValues[i] = key;
                                         distinctKeys[i].Add(key);
                                     }

                                     probeRows.Add((rightTuple, keyValues));
                                     if (probeRows.Count > TinyProbeUpperBound)
                                     {
                                         probeTooLarge = true;
                                         break;
                                     }
                                 }

                                 if (!probeTooLarge)
                                 {
                                     trace?.RecordStrategy(join, "KeyJoinScanIndexPartial");
                                     if (probeRows.Count == 0)
                                     {
                                         yield break;
                                     }

                                     var pivotPos = 0;
                                     var bestDistinct = distinctKeys[0].Count;
                                     var bestCached = context.FactIndices.ContainsKey((leftScanPredicate, join.LeftKeys[0]));
                                     for (var i = 1; i < joinKeyCount; i++)
                                     {
                                         var candidateDistinct = distinctKeys[i].Count;
                                         var candidateCached = context.FactIndices.ContainsKey((leftScanPredicate, join.LeftKeys[i]));

                                         if (candidateCached && !bestCached)
                                         {
                                             pivotPos = i;
                                             bestDistinct = candidateDistinct;
                                             bestCached = true;
                                             continue;
                                         }

                                         if (candidateCached == bestCached && candidateDistinct > bestDistinct)
                                         {
                                             pivotPos = i;
                                             bestDistinct = candidateDistinct;
                                         }
                                     }

                                     var pivotLeftKey = join.LeftKeys[pivotPos];

                                     var probeIndex = new Dictionary<object, List<(object[] Tuple, object[] Keys)>>();
                                     foreach (var probeRow in probeRows)
                                     {
                                         var pivotKey = probeRow.Keys[pivotPos];
                                         if (!probeIndex.TryGetValue(pivotKey, out var bucket))
                                         {
                                             bucket = new List<(object[] Tuple, object[] Keys)>();
                                             probeIndex[pivotKey] = bucket;
                                         }

                                         bucket.Add(probeRow);
                                     }

                                     if (probeIndex.Count == 0)
                                     {
                                         yield break;
                                     }

                                     var factIndex = GetFactIndex(leftScanPredicate, pivotLeftKey, facts, context);
                                     foreach (var entry in probeIndex)
                                     {
                                         if (!factIndex.TryGetValue(entry.Key, out var leftBucket))
                                         {
                                             continue;
                                         }

                                         foreach (var leftTuple in leftBucket)
                                         {
                                             if (leftTuple is null) continue;
                                             if (leftScanFilter is not null && !leftScanFilter(leftTuple)) continue;
                                             if (leftScanPattern is not null && !TupleMatchesPattern(leftTuple, leftScanPattern)) continue;

                                             foreach (var probeRow in entry.Value)
                                             {
                                                 var match = true;
                                                 for (var i = 0; i < joinKeyCount; i++)
                                                 {
                                                     if (i == pivotPos) continue;
                                                     var leftValue = GetLookupKey(leftTuple, join.LeftKeys[i], NullFactIndexKey);
                                                     if (!Equals(leftValue, probeRow.Keys[i]))
                                                     {
                                                         match = false;
                                                         break;
                                                     }
                                                 }

                                                 if (!match)
                                                 {
                                                     continue;
                                                 }

                                                 yield return BuildJoinOutput(leftTuple, probeRow.Tuple, join.LeftWidth, join.RightWidth, join.Width);
                                             }
                                         }
                                     }

                                     yield break;
                                 }

                                 trace?.RecordStrategy(join, "KeyJoinScanIndex");
                                 var fallbackIndex = GetJoinIndex(leftScanPredicate, leftIndexKeys, facts, context);
                                 foreach (var probeRow in probeRows)
                                 {
                                     var rightTuple = probeRow.Tuple;
                                     if (rightTuple is null) continue;

                                     var key = leftIndexKeys.Count == keyCount
                                         ? BuildKeyFromTuple(rightTuple, join.RightKeys)
                                         : BuildScanJoinLookupKey(rightTuple, join.RightKeys, join.LeftKeys, leftIndexKeys, leftScanPattern);
                                     var wrapper = new RowWrapper(key);

                                     if (!fallbackIndex.TryGetValue(wrapper, out var bucket))
                                     {
                                         continue;
                                     }

                                     foreach (var leftTuple in bucket)
                                     {
                                         if (leftTuple is null) continue;
                                         if (leftScanFilter is not null && !leftScanFilter(leftTuple)) continue;
                                         yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                     }
                                 }

                                 while (probeEnumerator.MoveNext())
                                 {
                                     var rightTuple = probeEnumerator.Current;
                                     if (rightTuple is null) continue;
                                     if (!ProbeMatchesScanJoinKeyConstants(rightTuple, join.RightKeys, join.LeftKeys, leftScanPattern)) continue;

                                     var key = leftIndexKeys.Count == keyCount
                                         ? BuildKeyFromTuple(rightTuple, join.RightKeys)
                                         : BuildScanJoinLookupKey(rightTuple, join.RightKeys, join.LeftKeys, leftIndexKeys, leftScanPattern);
                                     var wrapper = new RowWrapper(key);

                                     if (!fallbackIndex.TryGetValue(wrapper, out var bucket))
                                     {
                                         continue;
                                     }

                                     foreach (var leftTuple in bucket)
                                     {
                                         if (leftTuple is null) continue;
                                         if (leftScanFilter is not null && !leftScanFilter(leftTuple)) continue;
                                         yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                     }
                                 }

                                 yield break;
                             }

                             trace?.RecordStrategy(join, "KeyJoinScanIndex");
                             var index = GetJoinIndex(leftScanPredicate, leftIndexKeys, facts, context);
                             var probe = Evaluate(join.Right, context);
                            foreach (var rightTuple in probe)
                            {
                                if (rightTuple is null) continue;
                                if (!ProbeMatchesScanJoinKeyConstants(rightTuple, join.RightKeys, join.LeftKeys, leftScanPattern)) continue;

                                var key = leftIndexKeys.Count == keyCount
                                    ? BuildKeyFromTuple(rightTuple, join.RightKeys)
                                    : BuildScanJoinLookupKey(rightTuple, join.RightKeys, join.LeftKeys, leftIndexKeys, leftScanPattern);
                                var wrapper = new RowWrapper(key);

                                if (!index.TryGetValue(wrapper, out var bucket))
                                {
                                    continue;
                                }

                                foreach (var leftTuple in bucket)
                                {
                                    if (leftTuple is null) continue;
                                    if (leftScanFilter is not null && !leftScanFilter(leftTuple)) continue;
                                    yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                }
                            }
                        }

                        yield break;
                    }
                }
            }

            if (context is not null && (join.Left is MaterializeNode || join.Right is MaterializeNode))
            {
                trace?.RecordStrategy(join, "KeyJoinMaterializeIndex");
                if (join.Left is MaterializeNode leftMaterialize && join.Right is MaterializeNode rightMaterialize)
                {
                    var leftRows = Evaluate(leftMaterialize, context);
                    var rightRows = Evaluate(rightMaterialize, context);
                    var leftMaterialized = leftRows as List<object[]> ?? leftRows.ToList();
                    var rightMaterialized = rightRows as List<object[]> ?? rightRows.ToList();

                    if (leftMaterialized.Count <= rightMaterialized.Count)
                    {
                        if (joinKeyCount == 1)
                        {
                            var index = GetMaterializeFactIndex(leftMaterialize.Id, join.LeftKeys[0], leftMaterialized, context);
                            foreach (var rightTuple in rightMaterialized)
                            {
                                if (rightTuple is null) continue;

                                var lookupKey = GetLookupKey(rightTuple, join.RightKeys[0], NullFactIndexKey);
                                if (!index.TryGetValue(lookupKey, out var bucket))
                                {
                                    continue;
                                }

                                foreach (var leftTuple in bucket)
                                {
                                    yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                }
                            }

                            yield break;
                        }

                        var joinIndex = GetMaterializeJoinIndex(leftMaterialize.Id, join.LeftKeys, leftMaterialized, context);
                        foreach (var rightTuple in rightMaterialized)
                        {
                            if (rightTuple is null) continue;

                            var key = BuildKeyFromTuple(rightTuple, join.RightKeys);
                            var wrapper = new RowWrapper(key);

                            if (!joinIndex.TryGetValue(wrapper, out var bucket))
                            {
                                continue;
                            }

                            foreach (var leftTuple in bucket)
                            {
                                yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                            }
                        }

                        yield break;
                    }

                    if (joinKeyCount == 1)
                    {
                        var index = GetMaterializeFactIndex(rightMaterialize.Id, join.RightKeys[0], rightMaterialized, context);
                        foreach (var leftTuple in leftMaterialized)
                        {
                            if (leftTuple is null) continue;

                            var lookupKey = GetLookupKey(leftTuple, join.LeftKeys[0], NullFactIndexKey);
                            if (!index.TryGetValue(lookupKey, out var bucket))
                            {
                                continue;
                            }

                            foreach (var rightTuple in bucket)
                            {
                                yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                            }
                        }

                        yield break;
                    }

                    var joinIndexFallback = GetMaterializeJoinIndex(rightMaterialize.Id, join.RightKeys, rightMaterialized, context);
                    foreach (var leftTuple in leftMaterialized)
                    {
                        if (leftTuple is null) continue;

                        var key = BuildKeyFromTuple(leftTuple, join.LeftKeys);
                        var wrapper = new RowWrapper(key);

                        if (!joinIndexFallback.TryGetValue(wrapper, out var bucket))
                        {
                            continue;
                        }

                        foreach (var rightTuple in bucket)
                        {
                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                        }
                    }

                    yield break;
                }

                if (join.Left is MaterializeNode leftOnlyMaterialize)
                {
                    var buildRows = Evaluate(leftOnlyMaterialize, context);
                    var buildMaterialized = buildRows as List<object[]> ?? buildRows.ToList();
                    var probe = Evaluate(join.Right, context);

                    if (joinKeyCount == 1)
                    {
                        var index = GetMaterializeFactIndex(leftOnlyMaterialize.Id, join.LeftKeys[0], buildMaterialized, context);
                        foreach (var rightTuple in probe)
                        {
                            if (rightTuple is null) continue;

                            var lookupKey = GetLookupKey(rightTuple, join.RightKeys[0], NullFactIndexKey);
                            if (!index.TryGetValue(lookupKey, out var bucket))
                            {
                                continue;
                            }

                            foreach (var leftTuple in bucket)
                            {
                                yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                            }
                        }

                        yield break;
                    }

                    var joinIndex = GetMaterializeJoinIndex(leftOnlyMaterialize.Id, join.LeftKeys, buildMaterialized, context);
                    foreach (var rightTuple in probe)
                    {
                        if (rightTuple is null) continue;

                        var key = BuildKeyFromTuple(rightTuple, join.RightKeys);
                        var wrapper = new RowWrapper(key);

                        if (!joinIndex.TryGetValue(wrapper, out var bucket))
                        {
                            continue;
                        }

                        foreach (var leftTuple in bucket)
                        {
                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                        }
                    }

                    yield break;
                }

                if (join.Right is MaterializeNode rightOnlyMaterialize)
                {
                    var buildRows = Evaluate(rightOnlyMaterialize, context);
                    var buildMaterialized = buildRows as List<object[]> ?? buildRows.ToList();
                    var probe = Evaluate(join.Left, context);

                    if (joinKeyCount == 1)
                    {
                        var index = GetMaterializeFactIndex(rightOnlyMaterialize.Id, join.RightKeys[0], buildMaterialized, context);
                        foreach (var leftTuple in probe)
                        {
                            if (leftTuple is null) continue;

                            var lookupKey = GetLookupKey(leftTuple, join.LeftKeys[0], NullFactIndexKey);
                            if (!index.TryGetValue(lookupKey, out var bucket))
                            {
                                continue;
                            }

                            foreach (var rightTuple in bucket)
                            {
                                yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                            }
                        }

                        yield break;
                    }

                    var joinIndex = GetMaterializeJoinIndex(rightOnlyMaterialize.Id, join.RightKeys, buildMaterialized, context);
                    foreach (var leftTuple in probe)
                    {
                        if (leftTuple is null) continue;

                        var key = BuildKeyFromTuple(leftTuple, join.LeftKeys);
                        var wrapper = new RowWrapper(key);

                        if (!joinIndex.TryGetValue(wrapper, out var bucket))
                        {
                            continue;
                        }

                        foreach (var rightTuple in bucket)
                        {
                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                        }
                    }

                    yield break;
                }
            }

            static int EstimateBuildCost(PlanNode node) => node switch
            {
                ParamSeedNode => 0,
                UnitNode => 1,
                EmptyNode => 0,
                LimitNode limit => Math.Min(Math.Max(0, limit.Count), EstimateBuildCost(limit.Input)),
                ProjectionNode projection => EstimateBuildCost(projection.Input),
                SelectionNode selection => EstimateBuildCost(selection.Input),
                ArithmeticNode arithmetic => EstimateBuildCost(arithmetic.Input),
                NegationNode negation => EstimateBuildCost(negation.Input),
                AggregateNode aggregate => EstimateBuildCost(aggregate.Input),
                AggregateSubplanNode aggregateSubplan => EstimateBuildCost(aggregateSubplan.Input),
                OrderByNode orderBy => EstimateBuildCost(orderBy.Input),
                OffsetNode offset => EstimateBuildCost(offset.Input),
                MaterializeNode => 5,
                RelationScanNode => 20,
                PatternScanNode => 20,
                RecursiveRefNode => 50,
                CrossRefNode => 50,
                FixpointNode => 100,
                MutualFixpointNode => 100,
                _ => 25,
            };

            var buildLeft = EstimateBuildCost(join.Left) < EstimateBuildCost(join.Right);
            if (forceBuildLeft is not null)
            {
                buildLeft = forceBuildLeft.Value;
            }
            else if (context is not null)
            {
                var leftHasEstimate = TryEstimateRowUpperBound(join.Left, context, out var leftUpperBound) &&
                                      leftUpperBound != int.MaxValue;
                var rightHasEstimate = TryEstimateRowUpperBound(join.Right, context, out var rightUpperBound) &&
                                       rightUpperBound != int.MaxValue;

                if (leftHasEstimate && rightHasEstimate)
                {
                    if (leftUpperBound < rightUpperBound)
                    {
                        buildLeft = true;
                    }
                    else if (rightUpperBound < leftUpperBound)
                    {
                        buildLeft = false;
                    }
                }
                else if (leftHasEstimate && !rightHasEstimate)
                {
                    buildLeft = true;
                }
                else if (!leftHasEstimate && rightHasEstimate)
                {
                    buildLeft = false;
                }
            }

            static bool TryGetRecursiveRows(
                PlanNode node,
                EvaluationContext context,
                out PredicateId predicate,
                out RecursiveRefKind kind,
                out IReadOnlyList<object[]> rows,
                out Func<object[], bool>? filter)
            {
                predicate = default;
                kind = default;
                rows = Array.Empty<object[]>();
                filter = null;

                while (true)
                {
                    switch (node)
                    {
                        case SelectionNode selection:
                        {
                            var selectionPredicate = selection.Predicate;
                            if (filter is null)
                            {
                                filter = selectionPredicate;
                            }
                            else
                            {
                                var outer = filter;
                                filter = tuple => outer(tuple) && selectionPredicate(tuple);
                            }

                            node = selection.Input;
                            continue;
                        }

                        case RecursiveRefNode recursive:
                            if (!recursive.Predicate.Equals(context.Current))
                            {
                                throw new NotSupportedException(
                                    $"Cross-predicate recursion is not supported (referenced {recursive.Predicate} while evaluating {context.Current}).");
                            }

                            predicate = recursive.Predicate;
                            kind = recursive.Kind;
                            break;

                        case CrossRefNode cross:
                            predicate = cross.Predicate;
                            kind = cross.Kind;
                            break;

                        default:
                            return false;
                    }

                    break;
                }

                var map = kind == RecursiveRefKind.Total ? context.Totals : context.Deltas;
                rows = map.TryGetValue(predicate, out var resolved) ? resolved : Array.Empty<object[]>();
                return true;
            }

            PredicateId leftPredicate = default;
            RecursiveRefKind leftKind = default;
            IReadOnlyList<object[]> leftRecursiveRows = Array.Empty<object[]>();
            Func<object[], bool>? leftRecursiveFilter = null;
            var leftIsRecursive = context is not null &&
                                  TryGetRecursiveRows(join.Left, context, out leftPredicate, out leftKind, out leftRecursiveRows, out leftRecursiveFilter);

            PredicateId rightPredicate = default;
            RecursiveRefKind rightKind = default;
            IReadOnlyList<object[]> rightRecursiveRows = Array.Empty<object[]>();
            Func<object[], bool>? rightRecursiveFilter = null;
            var rightIsRecursive = context is not null &&
                                   TryGetRecursiveRows(join.Right, context, out rightPredicate, out rightKind, out rightRecursiveRows, out rightRecursiveFilter);

            if (context is not null && (leftIsRecursive || rightIsRecursive))
            {
                if (leftIsRecursive && rightIsRecursive)
                {
                    if (leftKind == RecursiveRefKind.Delta && rightKind != RecursiveRefKind.Delta)
                    {
                        buildLeft = false;
                    }
                    else if (rightKind == RecursiveRefKind.Delta && leftKind != RecursiveRefKind.Delta)
                    {
                        buildLeft = true;
                    }
                    else
                    {
                        buildLeft = leftRecursiveRows.Count <= rightRecursiveRows.Count;
                    }
                }
                else if (leftIsRecursive && leftKind == RecursiveRefKind.Delta)
                {
                    buildLeft = true;
                }
                else if (rightIsRecursive && rightKind == RecursiveRefKind.Delta)
                {
                    buildLeft = false;
                }
            }

            if (context is not null && buildLeft && leftIsRecursive)
            {
                if (joinKeyCount == 1)
                {
                    trace?.RecordStrategy(join, "KeyJoinRecursiveIndexLeft");
                    var probeTuples = Evaluate(join.Right, context);
                    var index = GetRecursiveFactIndex(leftPredicate, leftKind, join.LeftKeys[0], leftRecursiveRows, context);
                    foreach (var rightTuple in probeTuples)
                    {
                        if (rightTuple is null) continue;

                        var lookupKey = GetLookupKey(rightTuple, join.RightKeys[0], NullFactIndexKey);
                        if (!index.TryGetValue(lookupKey, out var bucket))
                        {
                            continue;
                        }

                        foreach (var leftTuple in bucket)
                        {
                            if (leftRecursiveFilter is not null && !leftRecursiveFilter(leftTuple)) continue;
                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                        }
                    }

                    yield break;
                }

                var signature = string.Join(",", join.LeftKeys);
                var joinIndexCached =
                    context.RecursiveJoinIndices.TryGetValue((leftPredicate, leftKind, signature), out var cachedJoinIndex) &&
                    ReferenceEquals(cachedJoinIndex.Rows, leftRecursiveRows);

                const int TinyProbeUpperBound = 64;
                if (!joinIndexCached)
                {
                    var probeSource = Evaluate(join.Right, context);
                    using var probeEnumerator = probeSource.GetEnumerator();
                    var probeRows = new List<(object[] Tuple, object[] Keys)>();
                    var distinctKeys = new HashSet<object>[joinKeyCount];
                    for (var i = 0; i < joinKeyCount; i++)
                    {
                        distinctKeys[i] = new HashSet<object>();
                    }

                    var probeTooLarge = false;
                    while (probeEnumerator.MoveNext())
                    {
                        var rightTuple = probeEnumerator.Current;
                        if (rightTuple is null) continue;

                        var keyValues = new object[joinKeyCount];
                        for (var i = 0; i < joinKeyCount; i++)
                        {
                            var key = GetLookupKey(rightTuple, join.RightKeys[i], NullFactIndexKey);
                            keyValues[i] = key;
                            distinctKeys[i].Add(key);
                        }

                        probeRows.Add((rightTuple, keyValues));
                        if (probeRows.Count > TinyProbeUpperBound)
                        {
                            probeTooLarge = true;
                            break;
                        }
                    }

                    if (!probeTooLarge)
                    {
                        trace?.RecordStrategy(join, "KeyJoinRecursiveIndexPartial");
                        if (probeRows.Count == 0)
                        {
                            yield break;
                        }

                        bool IsFactIndexCached(int columnIndex) =>
                            context.RecursiveFactIndices.TryGetValue((leftPredicate, leftKind, columnIndex), out var cache) &&
                            ReferenceEquals(cache.Rows, leftRecursiveRows);

                        var pivotPos = 0;
                        var bestDistinct = distinctKeys[0].Count;
                        var bestCached = IsFactIndexCached(join.LeftKeys[0]);
                        for (var i = 1; i < joinKeyCount; i++)
                        {
                            var candidateDistinct = distinctKeys[i].Count;
                            var candidateCached = IsFactIndexCached(join.LeftKeys[i]);

                            if (candidateCached && !bestCached)
                            {
                                pivotPos = i;
                                bestDistinct = candidateDistinct;
                                bestCached = true;
                                continue;
                            }

                            if (candidateCached == bestCached && candidateDistinct > bestDistinct)
                            {
                                pivotPos = i;
                                bestDistinct = candidateDistinct;
                            }
                        }

                        var pivotLeftKey = join.LeftKeys[pivotPos];

                        var probeIndex = new Dictionary<object, List<(object[] Tuple, object[] Keys)>>();
                        foreach (var probeRow in probeRows)
                        {
                            var pivotKey = probeRow.Keys[pivotPos];
                            if (!probeIndex.TryGetValue(pivotKey, out var bucket))
                            {
                                bucket = new List<(object[] Tuple, object[] Keys)>();
                                probeIndex[pivotKey] = bucket;
                            }

                            bucket.Add(probeRow);
                        }

                        if (probeIndex.Count == 0)
                        {
                            yield break;
                        }

                        var factIndex = GetRecursiveFactIndex(leftPredicate, leftKind, pivotLeftKey, leftRecursiveRows, context);
                        foreach (var entry in probeIndex)
                        {
                            if (!factIndex.TryGetValue(entry.Key, out var leftBucket))
                            {
                                continue;
                            }

                            foreach (var leftTuple in leftBucket)
                            {
                                if (leftTuple is null) continue;
                                if (leftRecursiveFilter is not null && !leftRecursiveFilter(leftTuple)) continue;

                                foreach (var probeRow in entry.Value)
                                {
                                    var match = true;
                                    for (var i = 0; i < joinKeyCount; i++)
                                    {
                                        if (i == pivotPos) continue;
                                        var leftValue = GetLookupKey(leftTuple, join.LeftKeys[i], NullFactIndexKey);
                                        if (!Equals(leftValue, probeRow.Keys[i]))
                                        {
                                            match = false;
                                            break;
                                        }
                                    }

                                    if (!match)
                                    {
                                        continue;
                                    }

                                    yield return BuildJoinOutput(leftTuple, probeRow.Tuple, join.LeftWidth, join.RightWidth, join.Width);
                                }
                            }
                        }

                        yield break;
                    }

                    trace?.RecordStrategy(join, "KeyJoinRecursiveIndexLeft");
                    var joinIndex = GetRecursiveJoinIndex(leftPredicate, leftKind, join.LeftKeys, leftRecursiveRows, context);
                    foreach (var probeRow in probeRows)
                    {
                        var rightTuple = probeRow.Tuple;
                        if (rightTuple is null) continue;

                        var key = BuildKeyFromTuple(rightTuple, join.RightKeys);
                        var wrapper = new RowWrapper(key);

                        if (!joinIndex.TryGetValue(wrapper, out var bucket))
                        {
                            continue;
                        }

                        foreach (var leftTuple in bucket)
                        {
                            if (leftRecursiveFilter is not null && !leftRecursiveFilter(leftTuple)) continue;
                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                        }
                    }

                    while (probeEnumerator.MoveNext())
                    {
                        var rightTuple = probeEnumerator.Current;
                        if (rightTuple is null) continue;

                        var key = BuildKeyFromTuple(rightTuple, join.RightKeys);
                        var wrapper = new RowWrapper(key);

                        if (!joinIndex.TryGetValue(wrapper, out var bucket))
                        {
                            continue;
                        }

                        foreach (var leftTuple in bucket)
                        {
                            if (leftRecursiveFilter is not null && !leftRecursiveFilter(leftTuple)) continue;
                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                        }
                    }

                    yield break;
                }

                trace?.RecordStrategy(join, "KeyJoinRecursiveIndexLeft");
                var probe = Evaluate(join.Right, context);
                var cachedIndex = GetRecursiveJoinIndex(leftPredicate, leftKind, join.LeftKeys, leftRecursiveRows, context);
                foreach (var rightTuple in probe)
                {
                    if (rightTuple is null) continue;

                    var key = BuildKeyFromTuple(rightTuple, join.RightKeys);
                    var wrapper = new RowWrapper(key);

                    if (!cachedIndex.TryGetValue(wrapper, out var bucket))
                    {
                        continue;
                    }

                    foreach (var leftTuple in bucket)
                    {
                        if (leftRecursiveFilter is not null && !leftRecursiveFilter(leftTuple)) continue;
                        yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                    }
                }

                yield break;
            }

            if (context is not null && !buildLeft && rightIsRecursive)
            {
                if (joinKeyCount == 1)
                {
                    trace?.RecordStrategy(join, "KeyJoinRecursiveIndexRight");
                    var probeTuples = Evaluate(join.Left, context);
                    var index = GetRecursiveFactIndex(rightPredicate, rightKind, join.RightKeys[0], rightRecursiveRows, context);
                    foreach (var leftTuple in probeTuples)
                    {
                        if (leftTuple is null) continue;

                        var lookupKey = GetLookupKey(leftTuple, join.LeftKeys[0], NullFactIndexKey);
                        if (!index.TryGetValue(lookupKey, out var bucket))
                        {
                            continue;
                        }

                        foreach (var rightTuple in bucket)
                        {
                            if (rightRecursiveFilter is not null && !rightRecursiveFilter(rightTuple)) continue;
                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                        }
                    }

                    yield break;
                }

                var signature = string.Join(",", join.RightKeys);
                var joinIndexCached =
                    context.RecursiveJoinIndices.TryGetValue((rightPredicate, rightKind, signature), out var cachedJoinIndex) &&
                    ReferenceEquals(cachedJoinIndex.Rows, rightRecursiveRows);

                const int TinyProbeUpperBound = 64;
                if (!joinIndexCached)
                {
                    var probeSource = Evaluate(join.Left, context);
                    using var probeEnumerator = probeSource.GetEnumerator();
                    var probeRows = new List<(object[] Tuple, object[] Keys)>();
                    var distinctKeys = new HashSet<object>[joinKeyCount];
                    for (var i = 0; i < joinKeyCount; i++)
                    {
                        distinctKeys[i] = new HashSet<object>();
                    }

                    var probeTooLarge = false;
                    while (probeEnumerator.MoveNext())
                    {
                        var leftTuple = probeEnumerator.Current;
                        if (leftTuple is null) continue;

                        var keyValues = new object[joinKeyCount];
                        for (var i = 0; i < joinKeyCount; i++)
                        {
                            var key = GetLookupKey(leftTuple, join.LeftKeys[i], NullFactIndexKey);
                            keyValues[i] = key;
                            distinctKeys[i].Add(key);
                        }

                        probeRows.Add((leftTuple, keyValues));
                        if (probeRows.Count > TinyProbeUpperBound)
                        {
                            probeTooLarge = true;
                            break;
                        }
                    }

                    if (!probeTooLarge)
                    {
                        trace?.RecordStrategy(join, "KeyJoinRecursiveIndexPartial");
                        if (probeRows.Count == 0)
                        {
                            yield break;
                        }

                        bool IsFactIndexCached(int columnIndex) =>
                            context.RecursiveFactIndices.TryGetValue((rightPredicate, rightKind, columnIndex), out var cache) &&
                            ReferenceEquals(cache.Rows, rightRecursiveRows);

                        var pivotPos = 0;
                        var bestDistinct = distinctKeys[0].Count;
                        var bestCached = IsFactIndexCached(join.RightKeys[0]);
                        for (var i = 1; i < joinKeyCount; i++)
                        {
                            var candidateDistinct = distinctKeys[i].Count;
                            var candidateCached = IsFactIndexCached(join.RightKeys[i]);

                            if (candidateCached && !bestCached)
                            {
                                pivotPos = i;
                                bestDistinct = candidateDistinct;
                                bestCached = true;
                                continue;
                            }

                            if (candidateCached == bestCached && candidateDistinct > bestDistinct)
                            {
                                pivotPos = i;
                                bestDistinct = candidateDistinct;
                            }
                        }

                        var pivotRightKey = join.RightKeys[pivotPos];

                        var probeIndex = new Dictionary<object, List<(object[] Tuple, object[] Keys)>>();
                        foreach (var probeRow in probeRows)
                        {
                            var pivotKey = probeRow.Keys[pivotPos];
                            if (!probeIndex.TryGetValue(pivotKey, out var bucket))
                            {
                                bucket = new List<(object[] Tuple, object[] Keys)>();
                                probeIndex[pivotKey] = bucket;
                            }

                            bucket.Add(probeRow);
                        }

                        if (probeIndex.Count == 0)
                        {
                            yield break;
                        }

                        var factIndex = GetRecursiveFactIndex(rightPredicate, rightKind, pivotRightKey, rightRecursiveRows, context);
                        foreach (var entry in probeIndex)
                        {
                            if (!factIndex.TryGetValue(entry.Key, out var rightBucket))
                            {
                                continue;
                            }

                            foreach (var rightTuple in rightBucket)
                            {
                                if (rightTuple is null) continue;
                                if (rightRecursiveFilter is not null && !rightRecursiveFilter(rightTuple)) continue;

                                foreach (var probeRow in entry.Value)
                                {
                                    var match = true;
                                    for (var i = 0; i < joinKeyCount; i++)
                                    {
                                        if (i == pivotPos) continue;
                                        var rightValue = GetLookupKey(rightTuple, join.RightKeys[i], NullFactIndexKey);
                                        if (!Equals(rightValue, probeRow.Keys[i]))
                                        {
                                            match = false;
                                            break;
                                        }
                                    }

                                    if (!match)
                                    {
                                        continue;
                                    }

                                    yield return BuildJoinOutput(probeRow.Tuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                }
                            }
                        }

                        yield break;
                    }

                    trace?.RecordStrategy(join, "KeyJoinRecursiveIndexRight");
                    var joinIndex = GetRecursiveJoinIndex(rightPredicate, rightKind, join.RightKeys, rightRecursiveRows, context);
                    foreach (var probeRow in probeRows)
                    {
                        var leftTuple = probeRow.Tuple;
                        if (leftTuple is null) continue;

                        var key = BuildKeyFromTuple(leftTuple, join.LeftKeys);
                        var wrapper = new RowWrapper(key);

                        if (!joinIndex.TryGetValue(wrapper, out var bucket))
                        {
                            continue;
                        }

                        foreach (var rightTuple in bucket)
                        {
                            if (rightRecursiveFilter is not null && !rightRecursiveFilter(rightTuple)) continue;
                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                        }
                    }

                    while (probeEnumerator.MoveNext())
                    {
                        var leftTuple = probeEnumerator.Current;
                        if (leftTuple is null) continue;

                        var key = BuildKeyFromTuple(leftTuple, join.LeftKeys);
                        var wrapper = new RowWrapper(key);

                        if (!joinIndex.TryGetValue(wrapper, out var bucket))
                        {
                            continue;
                        }

                        foreach (var rightTuple in bucket)
                        {
                            if (rightRecursiveFilter is not null && !rightRecursiveFilter(rightTuple)) continue;
                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                        }
                    }

                    yield break;
                }

                trace?.RecordStrategy(join, "KeyJoinRecursiveIndexRight");
                var probe = Evaluate(join.Left, context);
                var cachedIndex = GetRecursiveJoinIndex(rightPredicate, rightKind, join.RightKeys, rightRecursiveRows, context);
                foreach (var leftTuple in probe)
                {
                    if (leftTuple is null) continue;

                    var key = BuildKeyFromTuple(leftTuple, join.LeftKeys);
                    var wrapper = new RowWrapper(key);

                    if (!cachedIndex.TryGetValue(wrapper, out var bucket))
                    {
                        continue;
                    }

                    foreach (var rightTuple in bucket)
                    {
                        if (rightRecursiveFilter is not null && !rightRecursiveFilter(rightTuple)) continue;
                        yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                    }
                }

                yield break;
            }

            if (context is not null)
            {
                if (buildLeft && join.Left is RelationScanNode leftScan)
                {
                    trace?.RecordStrategy(join, "KeyJoinScanIndexReuse");
                    var facts = GetFactsList(leftScan.Relation, context);
                    if (joinKeyCount == 1)
                    {
                        var index = GetFactIndex(leftScan.Relation, join.LeftKeys[0], facts, context);
                        var probe = Evaluate(join.Right, context);
                        foreach (var rightTuple in probe)
                        {
                            if (rightTuple is null) continue;

                            var lookupKey = GetLookupKey(rightTuple, join.RightKeys[0], NullFactIndexKey);
                            if (!index.TryGetValue(lookupKey, out var bucket))
                            {
                                continue;
                            }

                            foreach (var leftTuple in bucket)
                            {
                                if (leftTuple is null) continue;
                                yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                            }
                        }

                        yield break;
                    }

                    var joinIndex = GetJoinIndex(leftScan.Relation, join.LeftKeys, facts, context);
                    var joinProbe = Evaluate(join.Right, context);
                    foreach (var rightTuple in joinProbe)
                    {
                        if (rightTuple is null) continue;

                        var key = BuildKeyFromTuple(rightTuple, join.RightKeys);
                        var wrapper = new RowWrapper(key);
                        if (!joinIndex.TryGetValue(wrapper, out var bucket))
                        {
                            continue;
                        }

                        foreach (var leftTuple in bucket)
                        {
                            if (leftTuple is null) continue;
                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                        }
                    }

                    yield break;
                }

                if (!buildLeft && join.Right is RelationScanNode rightScan)
                {
                    trace?.RecordStrategy(join, "KeyJoinScanIndexReuse");
                    var facts = GetFactsList(rightScan.Relation, context);
                    if (joinKeyCount == 1)
                    {
                        var index = GetFactIndex(rightScan.Relation, join.RightKeys[0], facts, context);
                        var probe = Evaluate(join.Left, context);
                        foreach (var leftTuple in probe)
                        {
                            if (leftTuple is null) continue;

                            var lookupKey = GetLookupKey(leftTuple, join.LeftKeys[0], NullFactIndexKey);
                            if (!index.TryGetValue(lookupKey, out var bucket))
                            {
                                continue;
                            }

                            foreach (var rightTuple in bucket)
                            {
                                if (rightTuple is null) continue;
                                yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                            }
                        }

                        yield break;
                    }

                    var joinIndex = GetJoinIndex(rightScan.Relation, join.RightKeys, facts, context);
                    var joinProbe = Evaluate(join.Left, context);
                    foreach (var leftTuple in joinProbe)
                    {
                        if (leftTuple is null) continue;

                        var key = BuildKeyFromTuple(leftTuple, join.LeftKeys);
                        var wrapper = new RowWrapper(key);
                        if (!joinIndex.TryGetValue(wrapper, out var bucket))
                        {
                            continue;
                        }

                        foreach (var rightTuple in bucket)
                        {
                            if (rightTuple is null) continue;
                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                        }
                    }

                    yield break;
                }
            }

            trace?.RecordStrategy(join, buildLeft ? "KeyJoinHashBuildLeft" : "KeyJoinHashBuildRight");
            var left = Evaluate(join.Left, context);
            var right = Evaluate(join.Right, context);

            if (joinKeyCount == 1)
            {
                if (buildLeft)
                {
                    var indexSingle = new Dictionary<object, List<object[]>>();

                    foreach (var leftTuple in left)
                    {
                        if (leftTuple is null) continue;

                        var key = GetLookupKey(leftTuple, join.LeftKeys[0], NullFactIndexKey);
                        if (!indexSingle.TryGetValue(key, out var bucket))
                        {
                            bucket = new List<object[]>();
                            indexSingle[key] = bucket;
                        }

                        bucket.Add(leftTuple);
                    }

                    foreach (var rightTuple in right)
                    {
                        if (rightTuple is null) continue;

                        var key = GetLookupKey(rightTuple, join.RightKeys[0], NullFactIndexKey);
                        if (!indexSingle.TryGetValue(key, out var bucket))
                        {
                            continue;
                        }

                        foreach (var leftTuple in bucket)
                        {
                            yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                        }
                    }

                    yield break;
                }

                var indexSingleFallback = new Dictionary<object, List<object[]>>();

                foreach (var rightTuple in right)
                {
                    if (rightTuple is null) continue;

                    var key = GetLookupKey(rightTuple, join.RightKeys[0], NullFactIndexKey);
                    if (!indexSingleFallback.TryGetValue(key, out var bucket))
                    {
                        bucket = new List<object[]>();
                        indexSingleFallback[key] = bucket;
                    }

                    bucket.Add(rightTuple);
                }

                foreach (var leftTuple in left)
                {
                    if (leftTuple is null) continue;

                    var key = GetLookupKey(leftTuple, join.LeftKeys[0], NullFactIndexKey);
                    if (!indexSingleFallback.TryGetValue(key, out var bucket))
                    {
                        continue;
                    }

                    foreach (var rightTuple in bucket)
                    {
                        yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                    }
                }

                yield break;
            }

            var indexFallback = new Dictionary<RowWrapper, List<object[]>>(new RowWrapperComparer(StructuralArrayComparer.Instance));

            if (buildLeft)
            {
                foreach (var leftTuple in left)
                {
                    if (leftTuple is null) continue;

                    var key = BuildKeyFromTuple(leftTuple, join.LeftKeys);
                    var wrapper = new RowWrapper(key);

                    if (!indexFallback.TryGetValue(wrapper, out var bucket))
                    {
                        bucket = new List<object[]>();
                        indexFallback[wrapper] = bucket;
                    }

                    bucket.Add(leftTuple);
                }

                foreach (var rightTuple in right)
                {
                    if (rightTuple is null) continue;

                    var key = BuildKeyFromTuple(rightTuple, join.RightKeys);
                    var wrapper = new RowWrapper(key);

                    if (!indexFallback.TryGetValue(wrapper, out var bucket))
                    {
                        continue;
                    }

                    foreach (var leftTuple in bucket)
                    {
                        yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                    }
                }

                yield break;
            }

            foreach (var rightTuple in right)
            {
                if (rightTuple is null) continue;

                var key = BuildKeyFromTuple(rightTuple, join.RightKeys);
                var wrapper = new RowWrapper(key);

                if (!indexFallback.TryGetValue(wrapper, out var bucket))
                {
                    bucket = new List<object[]>();
                    indexFallback[wrapper] = bucket;
                }

                bucket.Add(rightTuple);
            }

            foreach (var leftTuple in left)
            {
                if (leftTuple is null) continue;

                var key = BuildKeyFromTuple(leftTuple, join.LeftKeys);
                var wrapper = new RowWrapper(key);

                if (!indexFallback.TryGetValue(wrapper, out var bucket))
                {
                    continue;
                }

                foreach (var rightTuple in bucket)
                {
                    yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                }
            }
        }

        private static object[] BuildJoinOutput(object[] leftTuple, object[] rightTuple, int leftWidth, int rightWidth, int width)
        {
            var output = new object[width];

            if (leftWidth > 0)
            {
                Array.Copy(leftTuple, output, Math.Min(leftWidth, leftTuple.Length));
            }

            if (rightWidth > 0)
            {
                Array.Copy(rightTuple, 0, output, leftWidth, Math.Min(rightWidth, rightTuple.Length));
            }

            return output;
        }

        private List<object[]> GetFactsList(PredicateId predicate, EvaluationContext? context)
        {
            if (context is null)
            {
                var source = _provider.GetFacts(predicate) ?? Enumerable.Empty<object[]>();
                return source as List<object[]> ?? source.ToList();
            }

            if (context.Facts.TryGetValue(predicate, out var cached))
            {
                var trace = context.Trace;
                if (trace is not null)
                {
                    trace.RecordCacheLookup("Facts", $"{predicate.Name}/{predicate.Arity}", hit: true, built: false);
                }

                return cached;
            }

            var missTrace = context.Trace;
            if (missTrace is not null)
            {
                missTrace.RecordCacheLookup("Facts", $"{predicate.Name}/{predicate.Arity}", hit: false, built: true);
            }

            var factsSource = _provider.GetFacts(predicate) ?? Enumerable.Empty<object[]>();
            var facts = factsSource as List<object[]> ?? factsSource.ToList();
            context.Facts[predicate] = facts;
            return facts;
        }

        private HashSet<object[]> GetFactsSet(PredicateId predicate, EvaluationContext? context)
        {
            if (context is null)
            {
                return new HashSet<object[]>(_provider.GetFacts(predicate) ?? Enumerable.Empty<object[]>(), StructuralArrayComparer.Instance);
            }

            if (context.FactSets.TryGetValue(predicate, out var cached))
            {
                var trace = context.Trace;
                if (trace is not null)
                {
                    trace.RecordCacheLookup("FactSet", $"{predicate.Name}/{predicate.Arity}", hit: true, built: false);
                }

                return cached;
            }

            var missTrace = context.Trace;
            if (missTrace is not null)
            {
                missTrace.RecordCacheLookup("FactSet", $"{predicate.Name}/{predicate.Arity}", hit: false, built: true);
            }

            var set = new HashSet<object[]>(GetFactsList(predicate, context), StructuralArrayComparer.Instance);
            context.FactSets[predicate] = set;
            return set;
        }

        private Dictionary<RowWrapper, List<object[]>> GetJoinIndex(
            PredicateId predicate,
            IReadOnlyList<int> keyIndices,
            IReadOnlyList<object[]> facts,
            EvaluationContext context)
        {
            var signature = string.Join(",", keyIndices);
            var cacheKey = (predicate, signature);

            if (context.JoinIndices.TryGetValue(cacheKey, out var cached))
            {
                var trace = context.Trace;
                if (trace is not null)
                {
                    trace.RecordCacheLookup("JoinIndex", $"{predicate.Name}/{predicate.Arity}:keys=[{signature}]", hit: true, built: false);
                }

                return cached;
            }

            var missTrace = context.Trace;
            if (missTrace is not null)
            {
                missTrace.RecordCacheLookup("JoinIndex", $"{predicate.Name}/{predicate.Arity}:keys=[{signature}]", hit: false, built: true);
            }

            var index = new Dictionary<RowWrapper, List<object[]>>(new RowWrapperComparer(StructuralArrayComparer.Instance));

            foreach (var tuple in facts)
            {
                if (tuple is null) continue;

                var key = BuildKeyFromTuple(tuple, keyIndices);
                var wrapper = new RowWrapper(key);

                if (!index.TryGetValue(wrapper, out var bucket))
                {
                    bucket = new List<object[]>();
                    index[wrapper] = bucket;
                }

                bucket.Add(tuple);
            }

            context.JoinIndices[cacheKey] = index;
            return index;
        }

        private Dictionary<RowWrapper, List<object[]>> GetMaterializeJoinIndex(
            string id,
            IReadOnlyList<int> keyIndices,
            IReadOnlyList<object[]> rows,
            EvaluationContext context)
        {
            var signature = string.Join(",", keyIndices);
            var cacheKey = (id, signature);

            if (context.MaterializeJoinIndices.TryGetValue(cacheKey, out var cached))
            {
                var trace = context.Trace;
                if (trace is not null)
                {
                    trace.RecordCacheLookup("MaterializeJoinIndex", $"{id}:keys=[{signature}]", hit: true, built: false);
                }

                return cached;
            }

            var missTrace = context.Trace;
            if (missTrace is not null)
            {
                missTrace.RecordCacheLookup("MaterializeJoinIndex", $"{id}:keys=[{signature}]", hit: false, built: true);
            }

            var index = new Dictionary<RowWrapper, List<object[]>>(new RowWrapperComparer(StructuralArrayComparer.Instance));

            foreach (var tuple in rows)
            {
                if (tuple is null) continue;

                var key = BuildKeyFromTuple(tuple, keyIndices);
                var wrapper = new RowWrapper(key);

                if (!index.TryGetValue(wrapper, out var bucket))
                {
                    bucket = new List<object[]>();
                    index[wrapper] = bucket;
                }

                bucket.Add(tuple);
            }

            context.MaterializeJoinIndices[cacheKey] = index;
            return index;
        }

        private sealed class IncrementalFactIndexCache
        {
            public IncrementalFactIndexCache(IReadOnlyList<object[]> rows)
            {
                Rows = rows ?? throw new ArgumentNullException(nameof(rows));
            }

            public IReadOnlyList<object[]> Rows { get; private set; }

            public int IndexedCount { get; private set; }

            public Dictionary<object, List<object[]>> Index { get; } = new();

            public void Reset(IReadOnlyList<object[]> rows)
            {
                Rows = rows ?? throw new ArgumentNullException(nameof(rows));
                IndexedCount = 0;
                Index.Clear();
            }

            public void EnsureIndexed(int columnIndex)
            {
                if (IndexedCount > Rows.Count)
                {
                    IndexedCount = 0;
                    Index.Clear();
                }

                for (var i = IndexedCount; i < Rows.Count; i++)
                {
                    var tuple = Rows[i];
                    if (tuple is null) continue;

                    var value = columnIndex >= 0 && columnIndex < tuple.Length
                        ? tuple[columnIndex]
                        : null;
                    var key = value ?? NullFactIndexKey;

                    if (!Index.TryGetValue(key, out var bucket))
                    {
                        bucket = new List<object[]>();
                        Index[key] = bucket;
                    }

                    bucket.Add(tuple);
                }

                IndexedCount = Rows.Count;
            }
        }

        private sealed class IncrementalJoinIndexCache
        {
            public IncrementalJoinIndexCache(IReadOnlyList<object[]> rows)
            {
                Rows = rows ?? throw new ArgumentNullException(nameof(rows));
            }

            public IReadOnlyList<object[]> Rows { get; private set; }

            public int IndexedCount { get; private set; }

            public Dictionary<RowWrapper, List<object[]>> Index { get; }
                = new(new RowWrapperComparer(StructuralArrayComparer.Instance));

            public void Reset(IReadOnlyList<object[]> rows)
            {
                Rows = rows ?? throw new ArgumentNullException(nameof(rows));
                IndexedCount = 0;
                Index.Clear();
            }

            public void EnsureIndexed(IReadOnlyList<int> keyIndices)
            {
                if (keyIndices is null) throw new ArgumentNullException(nameof(keyIndices));

                if (IndexedCount > Rows.Count)
                {
                    IndexedCount = 0;
                    Index.Clear();
                }

                for (var i = IndexedCount; i < Rows.Count; i++)
                {
                    var tuple = Rows[i];
                    if (tuple is null) continue;

                    var key = BuildKeyFromTuple(tuple, keyIndices);
                    var wrapper = new RowWrapper(key);

                    if (!Index.TryGetValue(wrapper, out var bucket))
                    {
                        bucket = new List<object[]>();
                        Index[wrapper] = bucket;
                    }

                    bucket.Add(tuple);
                }

                IndexedCount = Rows.Count;
            }
        }

        private Dictionary<object, List<object[]>> GetRecursiveFactIndex(
            PredicateId predicate,
            RecursiveRefKind kind,
            int columnIndex,
            IReadOnlyList<object[]> rows,
            EvaluationContext context)
        {
            var cacheKey = (predicate, kind, columnIndex);
            if (!context.RecursiveFactIndices.TryGetValue(cacheKey, out var cache))
            {
                cache = new IncrementalFactIndexCache(rows);
                context.RecursiveFactIndices[cacheKey] = cache;
                var trace = context.Trace;
                if (trace is not null)
                {
                    trace.RecordCacheLookup("RecursiveFactIndex", $"{predicate.Name}/{predicate.Arity}:{kind}:col={columnIndex}", hit: false, built: true);
                }
            }
            else if (!ReferenceEquals(cache.Rows, rows))
            {
                cache.Reset(rows);
                var trace = context.Trace;
                if (trace is not null)
                {
                    trace.RecordCacheLookup("RecursiveFactIndex", $"{predicate.Name}/{predicate.Arity}:{kind}:col={columnIndex}", hit: false, built: true);
                }
            }
            else
            {
                var trace = context.Trace;
                if (trace is not null)
                {
                    trace.RecordCacheLookup("RecursiveFactIndex", $"{predicate.Name}/{predicate.Arity}:{kind}:col={columnIndex}", hit: true, built: false);
                }
            }

            cache.EnsureIndexed(columnIndex);
            return cache.Index;
        }

        private Dictionary<RowWrapper, List<object[]>> GetRecursiveJoinIndex(
            PredicateId predicate,
            RecursiveRefKind kind,
            IReadOnlyList<int> keyIndices,
            IReadOnlyList<object[]> rows,
            EvaluationContext context)
        {
            var signature = string.Join(",", keyIndices);
            var cacheKey = (predicate, kind, signature);

            if (!context.RecursiveJoinIndices.TryGetValue(cacheKey, out var cache))
            {
                cache = new IncrementalJoinIndexCache(rows);
                context.RecursiveJoinIndices[cacheKey] = cache;
                var trace = context.Trace;
                if (trace is not null)
                {
                    trace.RecordCacheLookup("RecursiveJoinIndex", $"{predicate.Name}/{predicate.Arity}:{kind}:keys=[{signature}]", hit: false, built: true);
                }
            }
            else if (!ReferenceEquals(cache.Rows, rows))
            {
                cache.Reset(rows);
                var trace = context.Trace;
                if (trace is not null)
                {
                    trace.RecordCacheLookup("RecursiveJoinIndex", $"{predicate.Name}/{predicate.Arity}:{kind}:keys=[{signature}]", hit: false, built: true);
                }
            }
            else
            {
                var trace = context.Trace;
                if (trace is not null)
                {
                    trace.RecordCacheLookup("RecursiveJoinIndex", $"{predicate.Name}/{predicate.Arity}:{kind}:keys=[{signature}]", hit: true, built: false);
                }
            }

            cache.EnsureIndexed(keyIndices);
            return cache.Index;
        }

        private Dictionary<object, List<object[]>> GetFactIndex(
            PredicateId predicate,
            int columnIndex,
            IReadOnlyList<object[]> facts,
            EvaluationContext context)
        {
            var cacheKey = (predicate, columnIndex);
            if (context.FactIndices.TryGetValue(cacheKey, out var cached))
            {
                var trace = context.Trace;
                if (trace is not null)
                {
                    trace.RecordCacheLookup("FactIndex", $"{predicate.Name}/{predicate.Arity}:col={columnIndex}", hit: true, built: false);
                }

                return cached;
            }

            var missTrace = context.Trace;
            if (missTrace is not null)
            {
                missTrace.RecordCacheLookup("FactIndex", $"{predicate.Name}/{predicate.Arity}:col={columnIndex}", hit: false, built: true);
            }

            var index = new Dictionary<object, List<object[]>>();

            foreach (var tuple in facts)
            {
                if (tuple is null) continue;

                var value = columnIndex >= 0 && columnIndex < tuple.Length
                    ? tuple[columnIndex]
                    : null;

                var key = value ?? NullFactIndexKey;

                if (!index.TryGetValue(key, out var bucket))
                {
                    bucket = new List<object[]>();
                    index[key] = bucket;
                }

                bucket.Add(tuple);
            }

            context.FactIndices[cacheKey] = index;
            return index;
        }

        private Dictionary<object, List<object[]>> GetMaterializeFactIndex(
            string id,
            int columnIndex,
            IReadOnlyList<object[]> rows,
            EvaluationContext context)
        {
            var cacheKey = (id, columnIndex);
            if (context.MaterializeFactIndices.TryGetValue(cacheKey, out var cached))
            {
                var trace = context.Trace;
                if (trace is not null)
                {
                    trace.RecordCacheLookup("MaterializeFactIndex", $"{id}:col={columnIndex}", hit: true, built: false);
                }

                return cached;
            }

            var missTrace = context.Trace;
            if (missTrace is not null)
            {
                missTrace.RecordCacheLookup("MaterializeFactIndex", $"{id}:col={columnIndex}", hit: false, built: true);
            }

            var index = new Dictionary<object, List<object[]>>();

            foreach (var tuple in rows)
            {
                if (tuple is null) continue;

                var value = columnIndex >= 0 && columnIndex < tuple.Length
                    ? tuple[columnIndex]
                    : null;

                var key = value ?? NullFactIndexKey;

                if (!index.TryGetValue(key, out var bucket))
                {
                    bucket = new List<object[]>();
                    index[key] = bucket;
                }

                bucket.Add(tuple);
            }

            context.MaterializeFactIndices[cacheKey] = index;
            return index;
        }

        private IReadOnlyList<object[]> SelectFactsForPattern(
            PredicateId predicate,
            IReadOnlyList<object[]> facts,
            object[] pattern,
            EvaluationContext? context)
        {
            if (context is null)
            {
                return facts;
            }

            var boundIndexCount = 0;
            for (var i = 0; i < pattern.Length; i++)
            {
                if (!ReferenceEquals(pattern[i], Wildcard.Value))
                {
                    boundIndexCount++;
                }
            }

            if (boundIndexCount == 0)
            {
                return facts;
            }

            if (boundIndexCount == 1)
            {
                for (var i = 0; i < pattern.Length; i++)
                {
                    var value = pattern[i];
                    if (ReferenceEquals(value, Wildcard.Value))
                    {
                        continue;
                    }

                    var singleFactIndex = GetFactIndex(predicate, i, facts, context);
                    var key = value ?? NullFactIndexKey;
                    return singleFactIndex.TryGetValue(key, out var bucket) ? bucket : Array.Empty<object[]>();
                }
            }

            var keyIndices = new List<int>(boundIndexCount);
            for (var i = 0; i < pattern.Length; i++)
            {
                if (!ReferenceEquals(pattern[i], Wildcard.Value))
                {
                    keyIndices.Add(i);
                }
            }

            var signature = string.Join(",", keyIndices);
            if (context.JoinIndices.TryGetValue((predicate, signature), out var cachedJoinIndex))
            {
                var joinKey = new object[keyIndices.Count];
                for (var i = 0; i < keyIndices.Count; i++)
                {
                    joinKey[i] = pattern[keyIndices[i]];
                }

                var wrapper = new RowWrapper(joinKey);
                return cachedJoinIndex.TryGetValue(wrapper, out var joinBucket) ? joinBucket : Array.Empty<object[]>();
            }

            var pivotIndex = keyIndices[0];
            var pivotKey = pattern[pivotIndex] ?? NullFactIndexKey;
            var bestBucketCount = int.MaxValue;
            foreach (var boundIndex in keyIndices)
            {
                if (!context.FactIndices.TryGetValue((predicate, boundIndex), out var cachedFactIndex))
                {
                    continue;
                }

                var key = pattern[boundIndex] ?? NullFactIndexKey;
                var candidate = cachedFactIndex.TryGetValue(key, out var bucket) ? bucket.Count : 0;
                if (candidate < bestBucketCount)
                {
                    bestBucketCount = candidate;
                    pivotIndex = boundIndex;
                    pivotKey = key;
                }
            }

            var factIndex = GetFactIndex(predicate, pivotIndex, facts, context);
            return factIndex.TryGetValue(pivotKey, out var pivotBucket) ? pivotBucket : Array.Empty<object[]>();
        }

        private IEnumerable<object[]> ExecuteNegation(NegationNode negation, EvaluationContext? context)
        {
            var input = Evaluate(negation.Input, context);
            var factSet = GetFactsSet(negation.Predicate, context);

            foreach (var tuple in input)
            {
                var key = negation.KeySelector(tuple);
                if (!factSet.Contains(key))
                {
                    yield return tuple;
                }
            }
        }

        private IEnumerable<object[]> ExecuteAggregate(AggregateNode aggregate, EvaluationContext? context)
        {
            if (aggregate is null) throw new ArgumentNullException(nameof(aggregate));

            var input = Evaluate(aggregate.Input, context);
            var facts = GetFactsList(aggregate.Predicate, context);
            var cache = new Dictionary<RowWrapper, List<object[]>>(new RowWrapperComparer(StructuralArrayComparer.Instance));
            var groupCount = aggregate.GroupByIndices?.Count ?? 0;
            var extensionSize = groupCount + 1;
            var inputWidth = aggregate.Width - extensionSize;

            foreach (var tuple in input)
            {
                if (tuple is null) continue;

                var pattern = aggregate.Pattern(tuple) ?? Array.Empty<object>();
                var wrapper = new RowWrapper(pattern);

                if (!cache.TryGetValue(wrapper, out var extensions))
                {
                    var candidates = SelectFactsForPattern(aggregate.Predicate, facts, pattern, context);
                    extensions = ComputeAggregateExtensions(candidates, pattern, aggregate);
                    cache[wrapper] = extensions;
                }

                foreach (var extension in extensions)
                {
                    var output = new object[aggregate.Width];
                    Array.Copy(tuple, output, Math.Min(tuple.Length, inputWidth));
                    Array.Copy(extension, 0, output, inputWidth, extension.Length);
                    yield return output;
                }
            }
        }

        private IEnumerable<object[]> ExecuteAggregateSubplan(AggregateSubplanNode aggregate, EvaluationContext? context)
        {
            if (aggregate is null) throw new ArgumentNullException(nameof(aggregate));

            var input = Evaluate(aggregate.Input, context);
            var cache = new Dictionary<RowWrapper, List<object[]>>(new RowWrapperComparer(StructuralArrayComparer.Instance));
            var groupCount = aggregate.GroupByIndices?.Count ?? 0;
            var extensionSize = groupCount + 1;
            var inputWidth = aggregate.Width - extensionSize;

            foreach (var tuple in input)
            {
                if (tuple is null) continue;

                var parameters = aggregate.ParameterSelector(tuple) ?? Array.Empty<object>();
                var wrapper = new RowWrapper(parameters);

                if (!cache.TryGetValue(wrapper, out var extensions))
                {
                    var innerContext = new EvaluationContext(new[] { parameters }, context);
                    var rows = Evaluate(aggregate.Subplan, innerContext).ToList();
                    extensions = ComputeAggregateExtensionsFromRows(rows, aggregate);
                    cache[wrapper] = extensions;
                }

                foreach (var extension in extensions)
                {
                    var output = new object[aggregate.Width];
                    Array.Copy(tuple, output, Math.Min(tuple.Length, inputWidth));
                    Array.Copy(extension, 0, output, inputWidth, extension.Length);
                    yield return output;
                }
            }
        }

        private static IEnumerable<object[]> EvaluateUnit(UnitNode unit)
        {
            if (unit is null) throw new ArgumentNullException(nameof(unit));
            yield return new object[unit.Width];
        }

        private static List<object[]> ComputeAggregateExtensionsFromRows(
            IReadOnlyList<object[]> rows,
            AggregateSubplanNode aggregate)
        {
            var groupCount = aggregate.GroupByIndices?.Count ?? 0;

            if (groupCount <= 0)
            {
                return ComputeGlobalAggregateFromRows(rows, aggregate);
            }

            return ComputeGroupedAggregateFromRows(rows, aggregate, groupCount);
        }

        private static List<object[]> ComputeAggregateExtensions(
            IReadOnlyList<object[]> facts,
            object[] pattern,
            AggregateNode aggregate)
        {
            var groupCount = aggregate.GroupByIndices?.Count ?? 0;

            if (groupCount <= 0)
            {
                return ComputeGlobalAggregate(facts, pattern, aggregate);
            }

            return ComputeGroupedAggregate(facts, pattern, aggregate, groupCount);
        }

        private static List<object[]> ComputeGlobalAggregateFromRows(
            IReadOnlyList<object[]> rows,
            AggregateSubplanNode aggregate)
        {
            var count = 0;
            var hasAny = false;
            decimal sum = 0;
            decimal? min = null;
            decimal? max = null;
            var bag = aggregate.Operation is AggregateOperation.Bag or AggregateOperation.Set ? new List<object?>() : null;
            HashSet<object?>? set = aggregate.Operation == AggregateOperation.Set ? new HashSet<object?>() : null;

            foreach (var row in rows)
            {
                if (row is null) continue;

                count++;

                switch (aggregate.Operation)
                {
                    case AggregateOperation.Count:
                        break;

                    case AggregateOperation.Sum:
                        sum += ConvertToDecimal(row, aggregate.ValueIndex);
                        hasAny = true;
                        break;

                    case AggregateOperation.Avg:
                        sum += ConvertToDecimal(row, aggregate.ValueIndex);
                        hasAny = true;
                        break;

                    case AggregateOperation.Min:
                    {
                        var value = ConvertToDecimal(row, aggregate.ValueIndex);
                        min = min is null ? value : Math.Min(min.Value, value);
                        hasAny = true;
                        break;
                    }

                    case AggregateOperation.Max:
                    {
                        var value = ConvertToDecimal(row, aggregate.ValueIndex);
                        max = max is null ? value : Math.Max(max.Value, value);
                        hasAny = true;
                        break;
                    }

                    case AggregateOperation.Set:
                    {
                        var value = GetFactValue(row, aggregate.ValueIndex);
                        set!.Add(value);
                        hasAny = true;
                        break;
                    }

                    case AggregateOperation.Bag:
                    {
                        var value = GetFactValue(row, aggregate.ValueIndex);
                        bag!.Add(value);
                        hasAny = true;
                        break;
                    }

                    default:
                        throw new NotSupportedException($"Unsupported aggregate operation: {aggregate.Operation}");
                }
            }

            if (aggregate.Operation == AggregateOperation.Count)
            {
                return new List<object[]> { new object[] { count } };
            }

            if (!hasAny)
            {
                return new List<object[]>();
            }

            return aggregate.Operation switch
            {
                AggregateOperation.Sum => new List<object[]> { new object[] { sum } },
                AggregateOperation.Avg => new List<object[]> { new object[] { sum / count } },
                AggregateOperation.Min => new List<object[]> { new object[] { min! } },
                AggregateOperation.Max => new List<object[]> { new object[] { max! } },
                AggregateOperation.Set => new List<object[]> { new object[] { set!.ToList() } },
                AggregateOperation.Bag => new List<object[]> { new object[] { bag! } },
                _ => throw new NotSupportedException($"Unsupported aggregate operation: {aggregate.Operation}")
            };
        }

        private static List<object[]> ComputeGlobalAggregate(
            IReadOnlyList<object[]> facts,
            object[] pattern,
            AggregateNode aggregate)
        {
            var count = 0;
            var hasAny = false;
            decimal sum = 0;
            decimal? min = null;
            decimal? max = null;
            var bag = aggregate.Operation is AggregateOperation.Bag or AggregateOperation.Set ? new List<object?>() : null;
            HashSet<object?>? set = aggregate.Operation == AggregateOperation.Set ? new HashSet<object?>() : null;

            foreach (var fact in facts)
            {
                if (fact is null) continue;
                if (!FactMatchesPattern(fact, pattern)) continue;

                count++;

                switch (aggregate.Operation)
                {
                    case AggregateOperation.Count:
                        break;

                    case AggregateOperation.Sum:
                        sum += ConvertToDecimal(fact, aggregate.ValueIndex);
                        hasAny = true;
                        break;

                    case AggregateOperation.Avg:
                        sum += ConvertToDecimal(fact, aggregate.ValueIndex);
                        hasAny = true;
                        break;

                    case AggregateOperation.Min:
                    {
                        var value = ConvertToDecimal(fact, aggregate.ValueIndex);
                        min = min is null ? value : Math.Min(min.Value, value);
                        hasAny = true;
                        break;
                    }

                    case AggregateOperation.Max:
                    {
                        var value = ConvertToDecimal(fact, aggregate.ValueIndex);
                        max = max is null ? value : Math.Max(max.Value, value);
                        hasAny = true;
                        break;
                    }

                    case AggregateOperation.Set:
                    {
                        var value = GetFactValue(fact, aggregate.ValueIndex);
                        set!.Add(value);
                        hasAny = true;
                        break;
                    }

                    case AggregateOperation.Bag:
                    {
                        var value = GetFactValue(fact, aggregate.ValueIndex);
                        bag!.Add(value);
                        hasAny = true;
                        break;
                    }

                    default:
                        throw new NotSupportedException($"Unsupported aggregate operation: {aggregate.Operation}");
                }
            }

            if (aggregate.Operation == AggregateOperation.Count)
            {
                return new List<object[]> { new object[] { count } };
            }

            if (!hasAny)
            {
                return new List<object[]>();
            }

            return aggregate.Operation switch
            {
                AggregateOperation.Sum => new List<object[]> { new object[] { sum } },
                AggregateOperation.Avg => new List<object[]> { new object[] { sum / count } },
                AggregateOperation.Min => new List<object[]> { new object[] { min! } },
                AggregateOperation.Max => new List<object[]> { new object[] { max! } },
                AggregateOperation.Set => new List<object[]> { new object[] { set!.ToList() } },
                AggregateOperation.Bag => new List<object[]> { new object[] { bag! } },
                _ => throw new NotSupportedException($"Unsupported aggregate operation: {aggregate.Operation}")
            };
        }

        private sealed class AggregateState
        {
            public required object[] Key { get; init; }
            public int Count { get; set; }
            public bool HasAny { get; set; }
            public decimal Sum { get; set; }
            public decimal? Min { get; set; }
            public decimal? Max { get; set; }
            public List<object?>? Bag { get; set; }
            public HashSet<object?>? Set { get; set; }
        }

        private static List<object[]> ComputeGroupedAggregate(
            IReadOnlyList<object[]> facts,
            object[] pattern,
            AggregateNode aggregate,
            int groupCount)
        {
            var groups = new Dictionary<RowWrapper, AggregateState>(new RowWrapperComparer(StructuralArrayComparer.Instance));

            foreach (var fact in facts)
            {
                if (fact is null) continue;
                if (!FactMatchesPattern(fact, pattern)) continue;

                var key = BuildGroupKey(fact, aggregate.GroupByIndices);
                var wrapper = new RowWrapper(key);

                if (!groups.TryGetValue(wrapper, out var state))
                {
                    state = new AggregateState
                    {
                        Key = key,
                        Bag = aggregate.Operation is AggregateOperation.Bag or AggregateOperation.Set ? new List<object?>() : null,
                        Set = aggregate.Operation == AggregateOperation.Set ? new HashSet<object?>() : null
                    };
                    groups[wrapper] = state;
                }

                state.Count++;

                switch (aggregate.Operation)
                {
                    case AggregateOperation.Count:
                        break;

                    case AggregateOperation.Sum:
                        state.Sum += ConvertToDecimal(fact, aggregate.ValueIndex);
                        state.HasAny = true;
                        break;

                    case AggregateOperation.Avg:
                        state.Sum += ConvertToDecimal(fact, aggregate.ValueIndex);
                        state.HasAny = true;
                        break;

                    case AggregateOperation.Min:
                    {
                        var value = ConvertToDecimal(fact, aggregate.ValueIndex);
                        state.Min = state.Min is null ? value : Math.Min(state.Min.Value, value);
                        state.HasAny = true;
                        break;
                    }

                    case AggregateOperation.Max:
                    {
                        var value = ConvertToDecimal(fact, aggregate.ValueIndex);
                        state.Max = state.Max is null ? value : Math.Max(state.Max.Value, value);
                        state.HasAny = true;
                        break;
                    }

                    case AggregateOperation.Set:
                    {
                        var value = GetFactValue(fact, aggregate.ValueIndex);
                        state.Set!.Add(value);
                        state.HasAny = true;
                        break;
                    }

                    case AggregateOperation.Bag:
                    {
                        var value = GetFactValue(fact, aggregate.ValueIndex);
                        state.Bag!.Add(value);
                        state.HasAny = true;
                        break;
                    }

                    default:
                        throw new NotSupportedException($"Unsupported aggregate operation: {aggregate.Operation}");
                }
            }

            var extensions = new List<object[]>();
            foreach (var state in groups.Values)
            {
                var extension = new object[groupCount + 1];
                Array.Copy(state.Key, extension, Math.Min(state.Key.Length, groupCount));

                switch (aggregate.Operation)
                {
                    case AggregateOperation.Count:
                        extension[groupCount] = state.Count;
                        extensions.Add(extension);
                        break;

                    case AggregateOperation.Sum:
                        if (state.HasAny)
                        {
                            extension[groupCount] = state.Sum;
                            extensions.Add(extension);
                        }
                        break;

                    case AggregateOperation.Avg:
                        if (state.HasAny)
                        {
                            extension[groupCount] = state.Sum / state.Count;
                            extensions.Add(extension);
                        }
                        break;

                    case AggregateOperation.Min:
                        if (state.HasAny)
                        {
                            extension[groupCount] = state.Min!;
                            extensions.Add(extension);
                        }
                        break;

                    case AggregateOperation.Max:
                        if (state.HasAny)
                        {
                            extension[groupCount] = state.Max!;
                            extensions.Add(extension);
                        }
                        break;

                    case AggregateOperation.Set:
                        if (state.HasAny)
                        {
                            extension[groupCount] = state.Set!.ToList();
                            extensions.Add(extension);
                        }
                        break;

                    case AggregateOperation.Bag:
                        if (state.HasAny)
                        {
                            extension[groupCount] = state.Bag!;
                            extensions.Add(extension);
                        }
                        break;

                    default:
                        throw new NotSupportedException($"Unsupported aggregate operation: {aggregate.Operation}");
                }
            }

            return extensions;
        }

        private static List<object[]> ComputeGroupedAggregateFromRows(
            IReadOnlyList<object[]> rows,
            AggregateSubplanNode aggregate,
            int groupCount)
        {
            var groups = new Dictionary<RowWrapper, AggregateState>(new RowWrapperComparer(StructuralArrayComparer.Instance));

            foreach (var row in rows)
            {
                if (row is null) continue;

                var key = BuildGroupKey(row, aggregate.GroupByIndices);
                var wrapper = new RowWrapper(key);

                if (!groups.TryGetValue(wrapper, out var state))
                {
                    state = new AggregateState
                    {
                        Key = key,
                        Bag = aggregate.Operation is AggregateOperation.Bag or AggregateOperation.Set ? new List<object?>() : null,
                        Set = aggregate.Operation == AggregateOperation.Set ? new HashSet<object?>() : null
                    };
                    groups[wrapper] = state;
                }

                state.Count++;

                switch (aggregate.Operation)
                {
                    case AggregateOperation.Count:
                        break;

                    case AggregateOperation.Sum:
                        state.Sum += ConvertToDecimal(row, aggregate.ValueIndex);
                        state.HasAny = true;
                        break;

                    case AggregateOperation.Avg:
                        state.Sum += ConvertToDecimal(row, aggregate.ValueIndex);
                        state.HasAny = true;
                        break;

                    case AggregateOperation.Min:
                    {
                        var value = ConvertToDecimal(row, aggregate.ValueIndex);
                        state.Min = state.Min is null ? value : Math.Min(state.Min.Value, value);
                        state.HasAny = true;
                        break;
                    }

                    case AggregateOperation.Max:
                    {
                        var value = ConvertToDecimal(row, aggregate.ValueIndex);
                        state.Max = state.Max is null ? value : Math.Max(state.Max.Value, value);
                        state.HasAny = true;
                        break;
                    }

                    case AggregateOperation.Set:
                    {
                        var value = GetFactValue(row, aggregate.ValueIndex);
                        state.Set!.Add(value);
                        state.HasAny = true;
                        break;
                    }

                    case AggregateOperation.Bag:
                    {
                        var value = GetFactValue(row, aggregate.ValueIndex);
                        state.Bag!.Add(value);
                        state.HasAny = true;
                        break;
                    }

                    default:
                        throw new NotSupportedException($"Unsupported aggregate operation: {aggregate.Operation}");
                }
            }

            var extensions = new List<object[]>();
            foreach (var state in groups.Values)
            {
                var extension = new object[groupCount + 1];
                Array.Copy(state.Key, extension, Math.Min(state.Key.Length, groupCount));

                switch (aggregate.Operation)
                {
                    case AggregateOperation.Count:
                        extension[groupCount] = state.Count;
                        extensions.Add(extension);
                        break;

                    case AggregateOperation.Sum:
                        if (state.HasAny)
                        {
                            extension[groupCount] = state.Sum;
                            extensions.Add(extension);
                        }
                        break;

                    case AggregateOperation.Avg:
                        if (state.HasAny)
                        {
                            extension[groupCount] = state.Sum / state.Count;
                            extensions.Add(extension);
                        }
                        break;

                    case AggregateOperation.Min:
                        if (state.HasAny)
                        {
                            extension[groupCount] = state.Min!;
                            extensions.Add(extension);
                        }
                        break;

                    case AggregateOperation.Max:
                        if (state.HasAny)
                        {
                            extension[groupCount] = state.Max!;
                            extensions.Add(extension);
                        }
                        break;

                    case AggregateOperation.Set:
                        if (state.HasAny)
                        {
                            extension[groupCount] = state.Set!.ToList();
                            extensions.Add(extension);
                        }
                        break;

                    case AggregateOperation.Bag:
                        if (state.HasAny)
                        {
                            extension[groupCount] = state.Bag!;
                            extensions.Add(extension);
                        }
                        break;

                    default:
                        throw new NotSupportedException($"Unsupported aggregate operation: {aggregate.Operation}");
                }
            }

            return extensions;
        }

        private static bool FactMatchesPattern(object[] fact, object[] pattern)
        {
            if (pattern.Length == 0)
            {
                return true;
            }

            if (fact.Length < pattern.Length)
            {
                return false;
            }

            for (var i = 0; i < pattern.Length; i++)
            {
                var expected = pattern[i];
                if (ReferenceEquals(expected, Wildcard.Value))
                {
                    continue;
                }

                if (!Equals(fact[i], expected))
                {
                    return false;
                }
            }

            return true;
        }

        private static object[] BuildGroupKey(object[] fact, IReadOnlyList<int> indices)
        {
            var key = new object[indices.Count];
            for (var i = 0; i < indices.Count; i++)
            {
                var idx = indices[i];
                key[i] = idx >= 0 && idx < fact.Length ? fact[idx] : null!;
            }
            return key;
        }

        private static object? GetFactValue(object[] fact, int index) =>
            index >= 0 && index < fact.Length ? fact[index] : null;

        private static decimal ConvertToDecimal(object[] fact, int index)
        {
            var value = GetFactValue(fact, index);
            if (value is decimal dec) return dec;
            if (value is int i) return i;
            if (value is long l) return l;
            if (value is double d) return (decimal)d;
            if (value is float f) return (decimal)f;
            if (value is string s) return decimal.Parse(s, NumberStyles.Any, CultureInfo.InvariantCulture);
            return Convert.ToDecimal(value, CultureInfo.InvariantCulture);
        }

        private IEnumerable<object[]> ExecuteUnion(UnionNode union, EvaluationContext? context) =>
            union.Sources.SelectMany(node => Evaluate(node, context));

        private IEnumerable<object[]> ExecuteArithmetic(ArithmeticNode arithmetic, EvaluationContext? context)
        {
            var input = Evaluate(arithmetic.Input, context);
            foreach (var tuple in input)
            {
                var result = new object[arithmetic.Width];
                Array.Copy(tuple, result, Math.Min(tuple.Length, arithmetic.Width));
                result[arithmetic.ResultIndex] = EvaluateArithmeticExpression(arithmetic.Expression, tuple);
                yield return result;
            }
        }

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

        private IEnumerable<object[]> ExecuteLimit(LimitNode limit, EvaluationContext? context)
        {
            if (limit is null) throw new ArgumentNullException(nameof(limit));
            if (limit.Count < 0) throw new ArgumentOutOfRangeException(nameof(limit), limit.Count, "Limit count must be non-negative.");
            return Evaluate(limit.Input, context).Take(limit.Count);
        }

        private IEnumerable<object[]> ExecuteOffset(OffsetNode offset, EvaluationContext? context)
        {
            if (offset is null) throw new ArgumentNullException(nameof(offset));
            if (offset.Count < 0) throw new ArgumentOutOfRangeException(nameof(offset), offset.Count, "Offset count must be non-negative.");
            return Evaluate(offset.Input, context).Skip(offset.Count);
        }

        private sealed class OrderByRowComparer : IComparer<object[]>
        {
            private readonly IReadOnlyList<OrderKey> _keys;

            public OrderByRowComparer(IReadOnlyList<OrderKey> keys)
            {
                _keys = keys;
            }

            public int Compare(object[]? x, object[]? y)
            {
                if (ReferenceEquals(x, y)) return 0;
                if (x is null) return -1;
                if (y is null) return 1;

                foreach (var key in _keys)
                {
                    var index = key.Index;
                    var left = index >= 0 && index < x.Length ? x[index] : null;
                    var right = index >= 0 && index < y.Length ? y[index] : null;
                    var cmp = CompareSortValues(left, right);
                    if (cmp == 0)
                    {
                        continue;
                    }

                    return key.Direction == OrderDirection.Desc ? -cmp : cmp;
                }

                if (x.Length != y.Length)
                {
                    return x.Length.CompareTo(y.Length);
                }

                for (var i = 0; i < x.Length; i++)
                {
                    var cmp = CompareSortValues(x[i], y[i]);
                    if (cmp != 0)
                    {
                        return cmp;
                    }
                }

                return 0;
            }

            private static int CompareSortValues(object? left, object? right)
            {
                if (ReferenceEquals(left, right)) return 0;
                if (left is null) return -1;
                if (right is null) return 1;

                var leftRank = GetSortRank(left);
                var rightRank = GetSortRank(right);
                if (leftRank != rightRank)
                {
                    return leftRank.CompareTo(rightRank);
                }

                switch (leftRank)
                {
                    case 1:
                        return ((bool)left).CompareTo((bool)right);
                    case 2:
                        if (TryConvertToDecimal(left, out var leftNumber) && TryConvertToDecimal(right, out var rightNumber))
                        {
                            return leftNumber.CompareTo(rightNumber);
                        }
                        break;
                    case 3:
                        return string.Compare((string)left, (string)right, StringComparison.Ordinal);
                    case 4:
                        return string.Compare(((JsonElement)left).ToString(), ((JsonElement)right).ToString(), StringComparison.Ordinal);
                }

                if (left is IComparable comparableLeft &&
                    right is IComparable &&
                    left.GetType() == right.GetType())
                {
                    return comparableLeft.CompareTo(right);
                }

                var typeCompare = string.Compare(left.GetType().FullName, right.GetType().FullName, StringComparison.Ordinal);
                if (typeCompare != 0)
                {
                    return typeCompare;
                }

                return string.Compare(
                    Convert.ToString(left, CultureInfo.InvariantCulture),
                    Convert.ToString(right, CultureInfo.InvariantCulture),
                    StringComparison.Ordinal);
            }

            private static int GetSortRank(object value) =>
                value switch
                {
                    bool => 1,
                    sbyte or byte or short or ushort or int or uint or long or ulong or float or double or decimal => 2,
                    string => 3,
                    JsonElement => 4,
                    _ => 5
                };

            private static bool TryConvertToDecimal(object value, out decimal number)
            {
                try
                {
                    number = Convert.ToDecimal(value, CultureInfo.InvariantCulture);
                    return true;
                }
                catch
                {
                    number = 0;
                    return false;
                }
            }
        }

        private IEnumerable<object[]> ExecuteOrderBy(OrderByNode orderBy, EvaluationContext? context)
        {
            if (orderBy is null) throw new ArgumentNullException(nameof(orderBy));
            if (orderBy.Keys is null || orderBy.Keys.Count == 0)
            {
                return Evaluate(orderBy.Input, context);
            }

            var rows = Evaluate(orderBy.Input, context).ToList();
            rows.Sort(new OrderByRowComparer(orderBy.Keys));
            return rows;
        }

        private IEnumerable<object[]> EvaluateParamSeed(ParamSeedNode seed, EvaluationContext? context)
        {
            var parameters = context?.Parameters ?? Enumerable.Empty<object[]>();

            foreach (var paramTuple in parameters)
            {
                if (paramTuple is null) continue;

                var tuple = new object[seed.Width];

                if (paramTuple.Length == seed.InputPositions.Count)
                {
                    for (var i = 0; i < seed.InputPositions.Count; i++)
                    {
                        tuple[seed.InputPositions[i]] = paramTuple[i];
                    }
                }
                else if (paramTuple.Length == seed.Width)
                {
                    foreach (var pos in seed.InputPositions)
                    {
                        tuple[pos] = paramTuple[pos];
                    }
                }
                else
                {
                    throw new InvalidOperationException(
                        $"Parameter tuple length {paramTuple.Length} does not match input positions ({seed.InputPositions.Count}) or arity ({seed.Width}).");
                }

                yield return tuple;
            }
        }

        private IEnumerable<object[]> EvaluateMaterialize(MaterializeNode node, EvaluationContext? context)
        {
            if (context is null)
            {
                return Evaluate(node.Plan, context).ToList();
            }

            if (context.Materialized.TryGetValue(node.Id, out var cached))
            {
                return cached;
            }

            var rows = Evaluate(node.Plan, context).ToList();
            context.Materialized[node.Id] = rows;
            return rows;
        }

        private static IEnumerable<object[]> FilterByParameters(
            IEnumerable<object[]> source,
            IReadOnlyList<int> inputPositions,
            IReadOnlyList<object[]> parameters)
        {
            if (inputPositions.Count == 0)
            {
                return source;
            }

            if (parameters.Count == 0)
            {
                return Enumerable.Empty<object[]>();
            }

            static object[] BuildParameterKey(object[] tuple, IReadOnlyList<int> positions)
            {
                if (tuple.Length == positions.Count)
                {
                    var key = new object[positions.Count];
                    Array.Copy(tuple, key, key.Length);
                    return key;
                }

                return BuildKeyFromTuple(tuple, positions);
            }

            var parameterSet = new HashSet<RowWrapper>(
                parameters.Where(p => p is not null).Select(p => new RowWrapper(BuildParameterKey(p, inputPositions))),
                new RowWrapperComparer(StructuralArrayComparer.Instance));

            return source.Where(tuple =>
            {
                if (tuple is null) return false;
                var key = BuildKeyFromTuple(tuple, inputPositions);
                return parameterSet.Contains(new RowWrapper(key));
            });
        }

        private IEnumerable<object[]> ExecuteBoundFactScan(
            RelationScanNode scan,
            IReadOnlyList<int> inputPositions,
            IReadOnlyList<object[]> parameters,
            EvaluationContext context)
        {
            if (scan is null) throw new ArgumentNullException(nameof(scan));
            if (inputPositions is null) throw new ArgumentNullException(nameof(inputPositions));
            if (parameters is null) throw new ArgumentNullException(nameof(parameters));
            if (context is null) throw new ArgumentNullException(nameof(context));

            var facts = GetFactsList(scan.Relation, context);
            if (inputPositions.Count == 0)
            {
                return facts;
            }

            if (parameters.Count == 0)
            {
                return Enumerable.Empty<object[]>();
            }

            if (inputPositions.Count == 1)
            {
                var columnIndex = inputPositions[0];
                var index = GetFactIndex(scan.Relation, columnIndex, facts, context);
                var keys = new HashSet<object>();

                foreach (var paramTuple in parameters)
                {
                    if (paramTuple is null) continue;
                    object? value = null;
                    if (paramTuple.Length == 1)
                    {
                        value = paramTuple[0];
                    }
                    else if (columnIndex >= 0 && columnIndex < paramTuple.Length)
                    {
                        value = paramTuple[columnIndex];
                    }

                    keys.Add(value ?? NullFactIndexKey);
                }

                return EnumerateBuckets(index, keys);
            }

            var joinIndex = GetJoinIndex(scan.Relation, inputPositions, facts, context);
            var keySet = new HashSet<RowWrapper>(new RowWrapperComparer(StructuralArrayComparer.Instance));

            foreach (var paramTuple in parameters)
            {
                if (paramTuple is null) continue;
                var key = paramTuple.Length == inputPositions.Count
                    ? paramTuple.ToArray()
                    : BuildKeyFromTuple(paramTuple, inputPositions);
                keySet.Add(new RowWrapper(key));
            }

            return EnumerateBuckets(joinIndex, keySet);
        }

        private static IEnumerable<object[]> EnumerateBuckets(
            Dictionary<object, List<object[]>> index,
            HashSet<object> keys)
        {
            foreach (var key in keys)
            {
                if (!index.TryGetValue(key, out var bucket))
                {
                    continue;
                }

                foreach (var tuple in bucket)
                {
                    yield return tuple;
                }
            }
        }

        private static IEnumerable<object[]> EnumerateBuckets(
            Dictionary<RowWrapper, List<object[]>> index,
            HashSet<RowWrapper> keys)
        {
            foreach (var key in keys)
            {
                if (!index.TryGetValue(key, out var bucket))
                {
                    continue;
                }

                foreach (var tuple in bucket)
                {
                    yield return tuple;
                }
            }
        }

        private static object[] BuildKeyFromTuple(object[] tuple, IReadOnlyList<int> inputPositions)
        {
            var key = new object[inputPositions.Count];

            if (tuple.Length == inputPositions.Count)
            {
                var fast = true;
                for (var i = 0; i < inputPositions.Count; i++)
                {
                    if (inputPositions[i] != i)
                    {
                        fast = false;
                        break;
                    }
                }

                if (fast)
                {
                    for (var i = 0; i < inputPositions.Count; i++)
                    {
                        key[i] = tuple[i];
                    }

                    return key;
                }
            }

            for (var i = 0; i < inputPositions.Count; i++)
            {
                var pos = inputPositions[i];
                key[i] = pos >= 0 && pos < tuple.Length ? tuple[pos] : null!;
            }

            return key;
        }

        private IEnumerable<object[]> ExecuteTransitiveClosure(TransitiveClosureNode closure, EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                trace?.RecordStrategy(closure, "TransitiveClosure");

                var predicate = closure.Predicate;
                var cacheKey = (closure.EdgeRelation, predicate);
                var traceKey = $"{predicate.Name}/{predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}";

                if (context.TransitiveClosureResults.TryGetValue(cacheKey, out var cachedRows))
                {
                    trace?.RecordCacheLookup("TransitiveClosure", traceKey, hit: true, built: false);
                    return cachedRows;
                }

                trace?.RecordCacheLookup("TransitiveClosure", traceKey, hit: false, built: true);

                var edges = GetFactsList(closure.EdgeRelation, context);
                var succIndex = GetFactIndex(closure.EdgeRelation, 0, edges, context);

                var visited = new HashSet<PairKey>();
                var totalRows = new List<object[]>();
                var delta = new List<PairKey>();

                foreach (var edge in edges)
                {
                    if (edge is null || edge.Length < 2)
                    {
                        continue;
                    }

                    var from = edge[0];
                    var to = edge[1];
                    var key = new PairKey(from, to);
                    if (visited.Add(key))
                    {
                        totalRows.Add(new object[] { from, to });
                        delta.Add(key);
                    }
                }

                var iteration = 0;
                trace?.RecordFixpointIteration(closure, predicate, iteration, delta.Count, totalRows.Count);

                while (delta.Count > 0)
                {
                    iteration++;
                    var nextDelta = new List<PairKey>();

                    foreach (var pair in delta)
                    {
                        var lookupKey = pair.To ?? NullFactIndexKey;
                        if (!succIndex.TryGetValue(lookupKey, out var bucket))
                        {
                            continue;
                        }

                        foreach (var edge in bucket)
                        {
                            if (edge is null || edge.Length < 2)
                            {
                                continue;
                            }

                            var next = edge[1];
                            var nextKey = new PairKey(pair.From, next);
                            if (visited.Add(nextKey))
                            {
                                totalRows.Add(new object[] { pair.From, next });
                                nextDelta.Add(nextKey);
                            }
                        }
                    }

                    delta = nextDelta;
                    trace?.RecordFixpointIteration(closure, predicate, iteration, delta.Count, totalRows.Count);
                }

                context.TransitiveClosureResults[cacheKey] = totalRows;
                return totalRows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private IEnumerable<object[]> ExecuteGroupedTransitiveClosure(GroupedTransitiveClosureNode closure, EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));
            if (closure.GroupIndices is null) throw new ArgumentNullException(nameof(closure.GroupIndices));

            var width = closure.Predicate.Arity;
            if (width < 2)
            {
                return Array.Empty<object[]>();
            }

            var groupCount = closure.GroupIndices.Count;
            if (groupCount == 0)
            {
                return ExecuteTransitiveClosure(new TransitiveClosureNode(closure.EdgeRelation, closure.Predicate), parentContext);
            }

            var maxRequiredIndex = ValidateGroupedClosureIndices(closure.GroupIndices, width);

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                trace?.RecordStrategy(closure, "GroupedTransitiveClosure");

                var predicate = closure.Predicate;
                var signature = string.Join(",", closure.GroupIndices);
                var cacheKey = (closure.EdgeRelation, predicate, signature);
                var traceKey = $"{predicate.Name}/{predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:groups=[{signature}]";

                if (context.GroupedTransitiveClosureResults.TryGetValue(cacheKey, out var cachedRows))
                {
                    trace?.RecordCacheLookup("GroupedTransitiveClosure", traceKey, hit: true, built: false);
                    return cachedRows;
                }

                trace?.RecordCacheLookup("GroupedTransitiveClosure", traceKey, hit: false, built: true);

                var edges = GetFactsList(closure.EdgeRelation, context);

                var edgeKeyIndices = new int[groupCount + 1];
                for (var i = 0; i < groupCount; i++)
                {
                    edgeKeyIndices[i] = closure.GroupIndices[i];
                }
                edgeKeyIndices[groupCount] = 0;

                var succIndex = GetJoinIndex(closure.EdgeRelation, edgeKeyIndices, edges, context);
                var wrapperComparer = new RowWrapperComparer(StructuralArrayComparer.Instance);

                var visited = new HashSet<RowWrapper>(wrapperComparer);
                var totalRows = new List<object[]>();
                var delta = new List<object[]>();

                foreach (var edge in edges)
                {
                    if (edge is null || edge.Length <= maxRequiredIndex)
                    {
                        continue;
                    }

                    var from = edge[0];
                    var to = edge[1];

                    var key = new object[groupCount + 2];
                    for (var i = 0; i < groupCount; i++)
                    {
                        key[i] = edge[closure.GroupIndices[i]];
                    }
                    key[groupCount] = from;
                    key[groupCount + 1] = to;

                    if (visited.Add(new RowWrapper(key)))
                    {
                        totalRows.Add(BuildGroupedClosureRow(width, groupCount, closure.GroupIndices, key));
                        delta.Add(key);
                    }
                }

                var iteration = 0;
                trace?.RecordFixpointIteration(closure, predicate, iteration, delta.Count, totalRows.Count);

                while (delta.Count > 0)
                {
                    iteration++;
                    var nextDelta = new List<object[]>();

                    foreach (var pair in delta)
                    {
                        var lookup = new object[groupCount + 1];
                        Array.Copy(pair, lookup, groupCount);
                        lookup[groupCount] = pair[groupCount + 1];

                        if (!succIndex.TryGetValue(new RowWrapper(lookup), out var bucket))
                        {
                            continue;
                        }

                        var from = pair[groupCount];
                        foreach (var edge in bucket)
                        {
                            if (edge is null || edge.Length <= maxRequiredIndex)
                            {
                                continue;
                            }

                            var next = edge[1];

                            var nextKey = new object[groupCount + 2];
                            Array.Copy(pair, nextKey, groupCount);
                            nextKey[groupCount] = from;
                            nextKey[groupCount + 1] = next;

                            if (visited.Add(new RowWrapper(nextKey)))
                            {
                                totalRows.Add(BuildGroupedClosureRow(width, groupCount, closure.GroupIndices, nextKey));
                                nextDelta.Add(nextKey);
                            }
                        }
                    }

                    delta = nextDelta;
                    trace?.RecordFixpointIteration(closure, predicate, iteration, delta.Count, totalRows.Count);
                }

                context.GroupedTransitiveClosureResults[cacheKey] = totalRows;
                return totalRows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private static object[] BuildGroupedClosureRow(
            int width,
            int groupCount,
            IReadOnlyList<int> groupIndices,
            object[] key)
        {
            var row = new object[width];
            row[0] = key[groupCount];
            row[1] = key[groupCount + 1];

            for (var i = 0; i < groupCount; i++)
            {
                row[groupIndices[i]] = key[i];
            }

            return row;
        }

        private static int ValidateGroupedClosureIndices(IReadOnlyList<int> groupIndices, int width)
        {
            var maxRequiredIndex = 1;
            var seen = new HashSet<int>();

            for (var i = 0; i < groupIndices.Count; i++)
            {
                var index = groupIndices[i];
                if (index < 0 || index >= width)
                {
                    throw new ArgumentOutOfRangeException(nameof(groupIndices), $"Group index {index} is outside predicate width {width}.");
                }

                if (index == 0 || index == 1)
                {
                    throw new ArgumentOutOfRangeException(nameof(groupIndices), $"Group index {index} overlaps closure endpoints.");
                }

                if (!seen.Add(index))
                {
                    throw new ArgumentOutOfRangeException(nameof(groupIndices), $"Duplicate group index {index}.");
                }

                if (index > maxRequiredIndex)
                {
                    maxRequiredIndex = index;
                }
            }

            return maxRequiredIndex;
        }

        private IEnumerable<object[]> ExecuteSeededGroupedTransitiveClosure(
            GroupedTransitiveClosureNode closure,
            IReadOnlyList<int> inputPositions,
            IReadOnlyList<object[]> parameters,
            EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));
            if (inputPositions is null) throw new ArgumentNullException(nameof(inputPositions));
            if (parameters is null) throw new ArgumentNullException(nameof(parameters));

            if (parameters.Count == 0)
            {
                return Enumerable.Empty<object[]>();
            }

            var hasSource = inputPositions.Contains(0);
            var hasTarget = inputPositions.Contains(1);

            if (hasSource && !hasTarget)
            {
                return ExecuteSeededGroupedTransitiveClosureBySource(closure, inputPositions, parameters, parentContext);
            }

            if (hasTarget && !hasSource)
            {
                return ExecuteSeededGroupedTransitiveClosureByTarget(closure, inputPositions, parameters, parentContext);
            }

            if (hasSource && hasTarget)
            {
                return ExecuteSeededGroupedTransitiveClosurePairs(closure, inputPositions, parameters, parentContext);
            }

            return FilterByParameters(ExecuteGroupedTransitiveClosure(closure, parentContext), inputPositions, parameters);
        }

        private IEnumerable<object[]> ExecuteSeededGroupedTransitiveClosurePairs(
            GroupedTransitiveClosureNode closure,
            IReadOnlyList<int> inputPositions,
            IReadOnlyList<object[]> parameters,
            EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));
            if (inputPositions is null) throw new ArgumentNullException(nameof(inputPositions));
            if (parameters is null) throw new ArgumentNullException(nameof(parameters));

            var groupCount = closure.GroupIndices.Count;
            if (groupCount == 0)
            {
                return ExecuteSeededTransitiveClosurePairs(new TransitiveClosureNode(closure.EdgeRelation, closure.Predicate), parameters, parentContext);
            }

            for (var i = 0; i < groupCount; i++)
            {
                if (!inputPositions.Contains(closure.GroupIndices[i]))
                {
                    var rows = ExecuteSeededGroupedTransitiveClosureBySource(closure, inputPositions, parameters, parentContext);
                    return FilterByParameters(rows, inputPositions, parameters);
                }
            }

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                var wrapperComparer = new RowWrapperComparer(StructuralArrayComparer.Instance);

                var targetsBySource = new Dictionary<RowWrapper, HashSet<object?>>(wrapperComparer);
                var sourcesByTarget = new Dictionary<RowWrapper, HashSet<object?>>(wrapperComparer);

                foreach (var paramTuple in parameters)
                {
                    if (paramTuple is null || paramTuple.Length == 0)
                    {
                        continue;
                    }

                    if (!TryGetParameterValue(paramTuple, inputPositions, 0, out var source))
                    {
                        continue;
                    }

                    if (!TryGetParameterValue(paramTuple, inputPositions, 1, out var target))
                    {
                        target = null;
                    }

                    var groupValues = new object[groupCount];
                    for (var i = 0; i < groupCount; i++)
                    {
                        var groupIndex = closure.GroupIndices[i];
                        if (!TryGetParameterValue(paramTuple, inputPositions, groupIndex, out var groupValue))
                        {
                            throw new InvalidOperationException($"Missing parameter value for group index {groupIndex}.");
                        }

                        groupValues[i] = groupValue!;
                    }

                    var sourceKey = new object[groupCount + 1];
                    Array.Copy(groupValues, 0, sourceKey, 0, groupCount);
                    sourceKey[groupCount] = source!;
                    var sourceWrapper = new RowWrapper(sourceKey);

                    if (!targetsBySource.TryGetValue(sourceWrapper, out var targetSet))
                    {
                        targetSet = new HashSet<object?>();
                        targetsBySource.Add(sourceWrapper, targetSet);
                    }

                    targetSet.Add(target);

                    var targetKey = new object[groupCount + 1];
                    Array.Copy(groupValues, 0, targetKey, 0, groupCount);
                    targetKey[groupCount] = target!;
                    var targetWrapper = new RowWrapper(targetKey);

                    if (!sourcesByTarget.TryGetValue(targetWrapper, out var sourceSet))
                    {
                        sourceSet = new HashSet<object?>();
                        sourcesByTarget.Add(targetWrapper, sourceSet);
                    }

                    sourceSet.Add(source);
                }

                trace?.RecordStrategy(closure, "GroupedTransitiveClosurePairs");

                const int MaxMemoizedGroupedPairSeeds = 32;
                var canMemoizePairs =
                    _cacheContext is not null &&
                    targetsBySource.Count <= MaxMemoizedGroupedPairSeeds &&
                    sourcesByTarget.Count <= MaxMemoizedGroupedPairSeeds;

                if (canMemoizePairs && targetsBySource.Count <= sourcesByTarget.Count)
                {
                    trace?.RecordStrategy(closure, "GroupedTransitiveClosurePairsMemoized");
                    trace?.RecordStrategy(closure, "GroupedTransitiveClosurePairsMemoizedForward");
                    return ExecuteSeededGroupedTransitiveClosurePairsMemoizedBySource(
                        closure,
                        inputPositions,
                        targetsBySource,
                        context);
                }

                if (canMemoizePairs)
                {
                    trace?.RecordStrategy(closure, "GroupedTransitiveClosurePairsMemoized");
                    trace?.RecordStrategy(closure, "GroupedTransitiveClosurePairsMemoizedBackward");
                    return ExecuteSeededGroupedTransitiveClosurePairsMemoizedByTarget(
                        closure,
                        inputPositions,
                        sourcesByTarget,
                        context);
                }

                if (targetsBySource.Count <= sourcesByTarget.Count)
                {
                    trace?.RecordStrategy(closure, "GroupedTransitiveClosurePairsForward");
                    return ExecuteSeededGroupedTransitiveClosurePairsForward(closure, targetsBySource, context);
                }

                trace?.RecordStrategy(closure, "GroupedTransitiveClosurePairsBackward");
                return ExecuteSeededGroupedTransitiveClosurePairsBackward(closure, sourcesByTarget, context);
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private IEnumerable<object[]> ExecuteSeededGroupedTransitiveClosureBySource(
            GroupedTransitiveClosureNode closure,
            IReadOnlyList<int> inputPositions,
            IReadOnlyList<object[]> parameters,
            EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));
            if (inputPositions is null) throw new ArgumentNullException(nameof(inputPositions));
            if (parameters is null) throw new ArgumentNullException(nameof(parameters));

            var width = closure.Predicate.Arity;
            if (width < 2)
            {
                return Array.Empty<object[]>();
            }

            var groupCount = closure.GroupIndices.Count;
            if (groupCount == 0)
            {
                return ExecuteSeededTransitiveClosure(new TransitiveClosureNode(closure.EdgeRelation, closure.Predicate), parameters, parentContext);
            }

            var maxRequiredIndex = ValidateGroupedClosureIndices(closure.GroupIndices, width);

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                trace?.RecordStrategy(closure, "GroupedTransitiveClosureSeeded");

                var predicate = closure.Predicate;
                var edges = GetFactsList(closure.EdgeRelation, context);

                var edgeKeyIndices = new int[groupCount + 1];
                for (var i = 0; i < groupCount; i++)
                {
                    edgeKeyIndices[i] = closure.GroupIndices[i];
                }
                edgeKeyIndices[groupCount] = 0;

                var succIndex = GetJoinIndex(closure.EdgeRelation, edgeKeyIndices, edges, context);
                var wrapperComparer = new RowWrapperComparer(StructuralArrayComparer.Instance);

                var visited = new HashSet<RowWrapper>(wrapperComparer);
                var seedKeys = new HashSet<RowWrapper>(wrapperComparer);
                var orderedSeeds = new List<object[]>();
                var totalRows = new List<object[]>();
                var delta = new List<object[]>();

                Dictionary<object, List<object[]>>? fromIndex = null;

                foreach (var paramTuple in parameters)
                {
                    if (paramTuple is null || paramTuple.Length == 0)
                    {
                        continue;
                    }

                    if (!TryGetParameterValue(paramTuple, inputPositions, 0, out var seed))
                    {
                        continue;
                    }

                    var filters = new object?[groupCount];
                    var allBound = true;
                    for (var i = 0; i < groupCount; i++)
                    {
                        var groupIndex = closure.GroupIndices[i];
                        if (inputPositions.Contains(groupIndex))
                        {
                            if (!TryGetParameterValue(paramTuple, inputPositions, groupIndex, out var filter))
                            {
                                throw new InvalidOperationException($"Missing parameter value for group index {groupIndex}.");
                            }

                            filters[i] = filter;
                        }
                        else
                        {
                            filters[i] = Wildcard.Value;
                            allBound = false;
                        }
                    }

                    var seedKey = new object[groupCount + 1];
                    for (var i = 0; i < groupCount; i++)
                    {
                        seedKey[i] = filters[i]!;
                    }
                    seedKey[groupCount] = seed!;

                    if (!seedKeys.Add(new RowWrapper(seedKey)))
                    {
                        continue;
                    }

                    orderedSeeds.Add(seedKey);

                    if (allBound)
                    {
                        var lookup = new object[groupCount + 1];
                        Array.Copy(seedKey, lookup, lookup.Length);

                        if (!succIndex.TryGetValue(new RowWrapper(lookup), out var bucket))
                        {
                            continue;
                        }

                        foreach (var edge in bucket)
                        {
                            if (edge is null || edge.Length <= maxRequiredIndex)
                            {
                                continue;
                            }

                            var to = edge[1];
                            var key = new object[groupCount + 2];
                            Array.Copy(seedKey, key, seedKey.Length);
                            key[groupCount + 1] = to;

                            if (visited.Add(new RowWrapper(key)))
                            {
                                totalRows.Add(BuildGroupedClosureRow(width, groupCount, closure.GroupIndices, key));
                                delta.Add(key);
                            }
                        }

                        continue;
                    }

                    fromIndex ??= GetFactIndex(closure.EdgeRelation, 0, edges, context);

                    var lookupKey = seed ?? NullFactIndexKey;
                    if (!fromIndex.TryGetValue(lookupKey, out var fromBucket))
                    {
                        continue;
                    }

                    foreach (var edge in fromBucket)
                    {
                        if (edge is null || edge.Length <= maxRequiredIndex)
                        {
                            continue;
                        }

                        if (!EdgeMatchesFilters(edge, closure.GroupIndices, filters))
                        {
                            continue;
                        }

                        var key = new object[groupCount + 2];
                        for (var i = 0; i < groupCount; i++)
                        {
                            key[i] = edge[closure.GroupIndices[i]];
                        }
                        key[groupCount] = seed!;
                        key[groupCount + 1] = edge[1];

                        if (visited.Add(new RowWrapper(key)))
                        {
                            totalRows.Add(BuildGroupedClosureRow(width, groupCount, closure.GroupIndices, key));
                            delta.Add(key);
                        }
                    }
                }

                orderedSeeds.Sort(CompareCacheSeedRows);
                var flatSeedKey = BuildFlatSeedCacheKey(orderedSeeds, groupCount + 1);
                var groupedKey = string.Join(",", closure.GroupIndices);
                var cacheKey = (closure.EdgeRelation, closure.Predicate, groupedKey);
                var traceKey =
                    $"{closure.Predicate.Name}/{closure.Predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:groups=[{groupedKey}]:seeds={orderedSeeds.Count}";

                if (context.GroupedTransitiveClosureSeededResults.TryGetValue(cacheKey, out var cachedBySeed) &&
                    cachedBySeed.TryGetValue(new RowWrapper(flatSeedKey), out var cachedRows))
                {
                    trace?.RecordCacheLookup("GroupedTransitiveClosureSeeded", traceKey, hit: true, built: false);
                    return cachedRows;
                }

                trace?.RecordCacheLookup("GroupedTransitiveClosureSeeded", traceKey, hit: false, built: true);

                var iteration = 0;
                trace?.RecordFixpointIteration(closure, predicate, iteration, delta.Count, totalRows.Count);

                while (delta.Count > 0)
                {
                    iteration++;
                    var nextDelta = new List<object[]>();

                    foreach (var pair in delta)
                    {
                        var lookup = new object[groupCount + 1];
                        Array.Copy(pair, lookup, groupCount);
                        lookup[groupCount] = pair[groupCount + 1];

                        if (!succIndex.TryGetValue(new RowWrapper(lookup), out var bucket))
                        {
                            continue;
                        }

                        var from = pair[groupCount];
                        foreach (var edge in bucket)
                        {
                            if (edge is null || edge.Length <= maxRequiredIndex)
                            {
                                continue;
                            }

                            var next = edge[1];

                            var nextKey = new object[groupCount + 2];
                            Array.Copy(pair, nextKey, groupCount);
                            nextKey[groupCount] = from;
                            nextKey[groupCount + 1] = next;

                            if (visited.Add(new RowWrapper(nextKey)))
                            {
                                totalRows.Add(BuildGroupedClosureRow(width, groupCount, closure.GroupIndices, nextKey));
                                nextDelta.Add(nextKey);
                            }
                        }
                    }

                    delta = nextDelta;
                    trace?.RecordFixpointIteration(closure, predicate, iteration, delta.Count, totalRows.Count);
                }

                if (!context.GroupedTransitiveClosureSeededResults.TryGetValue(cacheKey, out var storeBySeed))
                {
                    storeBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                    context.GroupedTransitiveClosureSeededResults.Add(cacheKey, storeBySeed);
                }

                storeBySeed[new RowWrapper(flatSeedKey)] = totalRows;
                return totalRows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private IEnumerable<object[]> ExecuteSeededGroupedTransitiveClosureByTarget(
            GroupedTransitiveClosureNode closure,
            IReadOnlyList<int> inputPositions,
            IReadOnlyList<object[]> parameters,
            EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));
            if (inputPositions is null) throw new ArgumentNullException(nameof(inputPositions));
            if (parameters is null) throw new ArgumentNullException(nameof(parameters));

            var width = closure.Predicate.Arity;
            if (width < 2)
            {
                return Array.Empty<object[]>();
            }

            var groupCount = closure.GroupIndices.Count;
            if (groupCount == 0)
            {
                return ExecuteSeededTransitiveClosureByTarget(new TransitiveClosureNode(closure.EdgeRelation, closure.Predicate), parameters, parentContext);
            }

            var maxRequiredIndex = ValidateGroupedClosureIndices(closure.GroupIndices, width);

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                trace?.RecordStrategy(closure, "GroupedTransitiveClosureSeededByTarget");

                var predicate = closure.Predicate;
                var edges = GetFactsList(closure.EdgeRelation, context);

                var edgeKeyIndices = new int[groupCount + 1];
                for (var i = 0; i < groupCount; i++)
                {
                    edgeKeyIndices[i] = closure.GroupIndices[i];
                }
                edgeKeyIndices[groupCount] = 1;

                var predIndex = GetJoinIndex(closure.EdgeRelation, edgeKeyIndices, edges, context);
                var wrapperComparer = new RowWrapperComparer(StructuralArrayComparer.Instance);

                var visited = new HashSet<RowWrapper>(wrapperComparer);
                var seedKeys = new HashSet<RowWrapper>(wrapperComparer);
                var orderedSeeds = new List<object[]>();
                var totalRows = new List<object[]>();
                var delta = new List<object[]>();

                Dictionary<object, List<object[]>>? toIndex = null;

                foreach (var paramTuple in parameters)
                {
                    if (paramTuple is null || paramTuple.Length == 0)
                    {
                        continue;
                    }

                    if (!TryGetParameterValue(paramTuple, inputPositions, 1, out var seed))
                    {
                        continue;
                    }

                    var filters = new object?[groupCount];
                    var allBound = true;
                    for (var i = 0; i < groupCount; i++)
                    {
                        var groupIndex = closure.GroupIndices[i];
                        if (inputPositions.Contains(groupIndex))
                        {
                            if (!TryGetParameterValue(paramTuple, inputPositions, groupIndex, out var filter))
                            {
                                throw new InvalidOperationException($"Missing parameter value for group index {groupIndex}.");
                            }

                            filters[i] = filter;
                        }
                        else
                        {
                            filters[i] = Wildcard.Value;
                            allBound = false;
                        }
                    }

                    var seedKey = new object[groupCount + 1];
                    for (var i = 0; i < groupCount; i++)
                    {
                        seedKey[i] = filters[i]!;
                    }
                    seedKey[groupCount] = seed!;

                    if (!seedKeys.Add(new RowWrapper(seedKey)))
                    {
                        continue;
                    }

                    orderedSeeds.Add(seedKey);

                    if (allBound)
                    {
                        var lookup = new object[groupCount + 1];
                        Array.Copy(seedKey, lookup, lookup.Length);

                        if (!predIndex.TryGetValue(new RowWrapper(lookup), out var bucket))
                        {
                            continue;
                        }

                        foreach (var edge in bucket)
                        {
                            if (edge is null || edge.Length <= maxRequiredIndex)
                            {
                                continue;
                            }

                            var from = edge[0];
                            var key = new object[groupCount + 2];
                            Array.Copy(seedKey, key, groupCount);
                            key[groupCount] = from;
                            key[groupCount + 1] = seed!;

                            if (visited.Add(new RowWrapper(key)))
                            {
                                totalRows.Add(BuildGroupedClosureRow(width, groupCount, closure.GroupIndices, key));
                                delta.Add(key);
                            }
                        }

                        continue;
                    }

                    toIndex ??= GetFactIndex(closure.EdgeRelation, 1, edges, context);

                    var lookupKey = seed ?? NullFactIndexKey;
                    if (!toIndex.TryGetValue(lookupKey, out var toBucket))
                    {
                        continue;
                    }

                    foreach (var edge in toBucket)
                    {
                        if (edge is null || edge.Length <= maxRequiredIndex)
                        {
                            continue;
                        }

                        if (!EdgeMatchesFilters(edge, closure.GroupIndices, filters))
                        {
                            continue;
                        }

                        var key = new object[groupCount + 2];
                        for (var i = 0; i < groupCount; i++)
                        {
                            key[i] = edge[closure.GroupIndices[i]];
                        }
                        key[groupCount] = edge[0];
                        key[groupCount + 1] = seed!;

                        if (visited.Add(new RowWrapper(key)))
                        {
                            totalRows.Add(BuildGroupedClosureRow(width, groupCount, closure.GroupIndices, key));
                            delta.Add(key);
                        }
                    }
                }

                orderedSeeds.Sort(CompareCacheSeedRows);
                var flatSeedKey = BuildFlatSeedCacheKey(orderedSeeds, groupCount + 1);
                var groupedKey = string.Join(",", closure.GroupIndices);
                var cacheKey = (closure.EdgeRelation, closure.Predicate, groupedKey);
                var traceKey =
                    $"{closure.Predicate.Name}/{closure.Predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:groups=[{groupedKey}]:seeds={orderedSeeds.Count}";

                if (context.GroupedTransitiveClosureSeededByTargetResults.TryGetValue(cacheKey, out var cachedBySeed) &&
                    cachedBySeed.TryGetValue(new RowWrapper(flatSeedKey), out var cachedRows))
                {
                    trace?.RecordCacheLookup("GroupedTransitiveClosureSeededByTarget", traceKey, hit: true, built: false);
                    return cachedRows;
                }

                trace?.RecordCacheLookup("GroupedTransitiveClosureSeededByTarget", traceKey, hit: false, built: true);

                var iteration = 0;
                trace?.RecordFixpointIteration(closure, predicate, iteration, delta.Count, totalRows.Count);

                while (delta.Count > 0)
                {
                    iteration++;
                    var nextDelta = new List<object[]>();

                    foreach (var pair in delta)
                    {
                        var lookup = new object[groupCount + 1];
                        Array.Copy(pair, lookup, groupCount);
                        lookup[groupCount] = pair[groupCount];

                        if (!predIndex.TryGetValue(new RowWrapper(lookup), out var bucket))
                        {
                            continue;
                        }

                        var target = pair[groupCount + 1];
                        foreach (var edge in bucket)
                        {
                            if (edge is null || edge.Length <= maxRequiredIndex)
                            {
                                continue;
                            }

                            var prev = edge[0];

                            var nextKey = new object[groupCount + 2];
                            Array.Copy(pair, nextKey, groupCount);
                            nextKey[groupCount] = prev;
                            nextKey[groupCount + 1] = target;

                            if (visited.Add(new RowWrapper(nextKey)))
                            {
                                totalRows.Add(BuildGroupedClosureRow(width, groupCount, closure.GroupIndices, nextKey));
                                nextDelta.Add(nextKey);
                            }
                        }
                    }

                    delta = nextDelta;
                    trace?.RecordFixpointIteration(closure, predicate, iteration, delta.Count, totalRows.Count);
                }

                if (!context.GroupedTransitiveClosureSeededByTargetResults.TryGetValue(cacheKey, out var storeBySeed))
                {
                    storeBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                    context.GroupedTransitiveClosureSeededByTargetResults.Add(cacheKey, storeBySeed);
                }

                storeBySeed[new RowWrapper(flatSeedKey)] = totalRows;
                return totalRows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private IEnumerable<object[]> ExecuteSeededGroupedTransitiveClosurePairsMemoizedBySource(
            GroupedTransitiveClosureNode closure,
            IReadOnlyList<int> inputPositions,
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> targetsBySource,
            EvaluationContext context)
        {
            var totalRows = new List<object[]>();
            var groupCount = closure.GroupIndices.Count;

            foreach (var entry in targetsBySource)
            {
                var seedKey = entry.Key.Row;
                if (seedKey.Length != groupCount + 1)
                {
                    continue;
                }

                var source = seedKey[groupCount];
                var targets = entry.Value;
                if (targets.Count == 0)
                {
                    continue;
                }

                var seedParams = new List<object[]>(1)
                {
                    BuildGroupedPairSeedTuple(
                        closure,
                        inputPositions,
                        seedKey,
                        seedPosition: 0,
                        other: targets.FirstOrDefault())
                };
                var reachableRows = ExecuteSeededGroupedTransitiveClosureBySource(closure, inputPositions, seedParams, context);

                if (targets.Contains(null))
                {
                    foreach (var row in reachableRows)
                    {
                        totalRows.Add(row);
                    }

                    continue;
                }

                foreach (var row in reachableRows)
                {
                    if (row is null || row.Length < 2)
                    {
                        continue;
                    }

                    if (Equals(row[0], source) && targets.Contains(row[1]))
                    {
                        totalRows.Add(row);
                    }
                }
            }

            return totalRows;
        }

        private IEnumerable<object[]> ExecuteSeededGroupedTransitiveClosurePairsMemoizedByTarget(
            GroupedTransitiveClosureNode closure,
            IReadOnlyList<int> inputPositions,
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> sourcesByTarget,
            EvaluationContext context)
        {
            var totalRows = new List<object[]>();
            var groupCount = closure.GroupIndices.Count;

            foreach (var entry in sourcesByTarget)
            {
                var seedKey = entry.Key.Row;
                if (seedKey.Length != groupCount + 1)
                {
                    continue;
                }

                var target = seedKey[groupCount];
                var sources = entry.Value;
                if (sources.Count == 0)
                {
                    continue;
                }

                var seedParams = new List<object[]>(1)
                {
                    BuildGroupedPairSeedTuple(
                        closure,
                        inputPositions,
                        seedKey,
                        seedPosition: 1,
                        other: sources.FirstOrDefault())
                };
                var reachableRows = ExecuteSeededGroupedTransitiveClosureByTarget(closure, inputPositions, seedParams, context);

                if (sources.Contains(null))
                {
                    foreach (var row in reachableRows)
                    {
                        totalRows.Add(row);
                    }

                    continue;
                }

                foreach (var row in reachableRows)
                {
                    if (row is null || row.Length < 2)
                    {
                        continue;
                    }

                    if (Equals(row[1], target) && sources.Contains(row[0]))
                    {
                        totalRows.Add(row);
                    }
                }
            }

            return totalRows;
        }

        private static object[] BuildGroupedPairSeedTuple(
            GroupedTransitiveClosureNode closure,
            IReadOnlyList<int> inputPositions,
            object[] key,
            int seedPosition,
            object? other)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));
            if (inputPositions is null) throw new ArgumentNullException(nameof(inputPositions));
            if (key is null) throw new ArgumentNullException(nameof(key));

            var groupCount = closure.GroupIndices.Count;
            if (key.Length != groupCount + 1)
            {
                throw new ArgumentException($"Expected grouped key width {groupCount + 1} but found {key.Length}.", nameof(key));
            }

            if (seedPosition != 0 && seedPosition != 1)
            {
                throw new ArgumentOutOfRangeException(nameof(seedPosition), "Seed position must be 0 or 1.");
            }

            var tuple = new object[inputPositions.Count];
            for (var i = 0; i < inputPositions.Count; i++)
            {
                var position = inputPositions[i];
                if (position == seedPosition)
                {
                    tuple[i] = key[groupCount];
                    continue;
                }

                if (position == (seedPosition == 0 ? 1 : 0))
                {
                    tuple[i] = other!;
                    continue;
                }

                var groupSlot = -1;
                for (var j = 0; j < groupCount; j++)
                {
                    if (closure.GroupIndices[j] == position)
                    {
                        groupSlot = j;
                        break;
                    }
                }

                tuple[i] = groupSlot >= 0 ? key[groupSlot] : null!;
            }

            return tuple;
        }

        private IEnumerable<object[]> ExecuteSeededGroupedTransitiveClosurePairsForward(
            GroupedTransitiveClosureNode closure,
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> targetsBySource,
            EvaluationContext context)
        {
            var width = closure.Predicate.Arity;
            var groupCount = closure.GroupIndices.Count;
            var maxRequiredIndex = ValidateGroupedClosureIndices(closure.GroupIndices, width);
            var trace = context.Trace;
            var predicate = closure.Predicate;
            var edges = GetFactsList(closure.EdgeRelation, context);

            var edgeKeyIndices = new int[groupCount + 1];
            for (var i = 0; i < groupCount; i++)
            {
                edgeKeyIndices[i] = closure.GroupIndices[i];
            }

            edgeKeyIndices[groupCount] = 0;
            var succIndex = GetJoinIndex(closure.EdgeRelation, edgeKeyIndices, edges, context);

            var totalRows = new List<object[]>();

            foreach (var entry in targetsBySource)
            {
                var seedKey = entry.Key.Row;
                if (seedKey.Length != groupCount + 1 || entry.Value.Count == 0)
                {
                    continue;
                }

                var seed = seedKey[groupCount];
                var targets = entry.Value;

                if (targets.Contains(null))
                {
                    var seedParams = new List<object[]>(1) { seedKey };
                    var rows = ExecuteSeededGroupedTransitiveClosureBySource(closure, closure.GroupIndices.Append(0).ToArray(), seedParams, context);
                    foreach (var row in rows)
                    {
                        totalRows.Add(row);
                    }

                    continue;
                }

                var remaining = new HashSet<object?>(targets);
                var visitedNodes = new HashSet<object?>();
                var queue = new Queue<object?>();

                SeedForwardSearch(seedKey, seed, remaining, succIndex, maxRequiredIndex, visitedNodes, queue, totalRows, width, groupCount, closure.GroupIndices);

                var iteration = 0;
                trace?.RecordFixpointIteration(closure, predicate, iteration, queue.Count, visitedNodes.Count);

                while (queue.Count > 0 && remaining.Count > 0)
                {
                    iteration++;
                    var breadth = queue.Count;
                    for (var i = 0; i < breadth; i++)
                    {
                        var node = queue.Dequeue();
                        ExpandForwardSearch(seedKey, seed, node, remaining, succIndex, maxRequiredIndex, visitedNodes, queue, totalRows, width, groupCount, closure.GroupIndices);
                        if (remaining.Count == 0)
                        {
                            break;
                        }
                    }

                    trace?.RecordFixpointIteration(closure, predicate, iteration, queue.Count, visitedNodes.Count);
                }
            }

            return totalRows;

            void SeedForwardSearch(
                object[] key,
                object? seed,
                HashSet<object?> remaining,
                Dictionary<RowWrapper, List<object[]>> index,
                int requiredEdgeWidth,
                HashSet<object?> visited,
                Queue<object?> frontier,
                List<object[]> output,
                int rowWidth,
                int groupWidth,
                IReadOnlyList<int> groupIndices)
            {
                visited.Add(seed);
                ExpandForwardSearch(key, seed, seed, remaining, index, requiredEdgeWidth, visited, frontier, output, rowWidth, groupWidth, groupIndices);
            }

            void ExpandForwardSearch(
                object[] key,
                object? seed,
                object? current,
                HashSet<object?> remaining,
                Dictionary<RowWrapper, List<object[]>> index,
                int requiredEdgeWidth,
                HashSet<object?> visited,
                Queue<object?> frontier,
                List<object[]> output,
                int rowWidth,
                int groupWidth,
                IReadOnlyList<int> groupIndices)
            {
                var lookup = new object[groupWidth + 1];
                Array.Copy(key, 0, lookup, 0, groupWidth);
                lookup[groupWidth] = current!;

                if (!index.TryGetValue(new RowWrapper(lookup), out var bucket))
                {
                    return;
                }

                foreach (var edge in bucket)
                {
                    if (edge is null || edge.Length <= requiredEdgeWidth)
                    {
                        continue;
                    }

                    var next = edge[1];
                    if (remaining.Remove(next))
                    {
                        var rowKey = new object[groupWidth + 2];
                        Array.Copy(key, 0, rowKey, 0, groupWidth);
                        rowKey[groupWidth] = seed!;
                        rowKey[groupWidth + 1] = next;
                        output.Add(BuildGroupedClosureRow(rowWidth, groupWidth, groupIndices, rowKey));
                        if (remaining.Count == 0)
                        {
                            return;
                        }
                    }

                    if (!visited.Add(next))
                    {
                        continue;
                    }

                    frontier.Enqueue(next);
                }
            }
        }

        private IEnumerable<object[]> ExecuteSeededGroupedTransitiveClosurePairsBackward(
            GroupedTransitiveClosureNode closure,
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> sourcesByTarget,
            EvaluationContext context)
        {
            var width = closure.Predicate.Arity;
            var groupCount = closure.GroupIndices.Count;
            var maxRequiredIndex = ValidateGroupedClosureIndices(closure.GroupIndices, width);
            var trace = context.Trace;
            var predicate = closure.Predicate;
            var edges = GetFactsList(closure.EdgeRelation, context);

            var edgeKeyIndices = new int[groupCount + 1];
            for (var i = 0; i < groupCount; i++)
            {
                edgeKeyIndices[i] = closure.GroupIndices[i];
            }

            edgeKeyIndices[groupCount] = 1;
            var predIndex = GetJoinIndex(closure.EdgeRelation, edgeKeyIndices, edges, context);

            var totalRows = new List<object[]>();

            foreach (var entry in sourcesByTarget)
            {
                var seedKey = entry.Key.Row;
                if (seedKey.Length != groupCount + 1 || entry.Value.Count == 0)
                {
                    continue;
                }

                var target = seedKey[groupCount];
                var sources = entry.Value;

                if (sources.Contains(null))
                {
                    var seedParams = new List<object[]>(1) { seedKey };
                    var rows = ExecuteSeededGroupedTransitiveClosureByTarget(closure, closure.GroupIndices.Append(1).ToArray(), seedParams, context);
                    foreach (var row in rows)
                    {
                        totalRows.Add(row);
                    }

                    continue;
                }

                var remaining = new HashSet<object?>(sources);
                var visitedNodes = new HashSet<object?>();
                var queue = new Queue<object?>();

                SeedBackwardSearch(seedKey, target, remaining, predIndex, maxRequiredIndex, visitedNodes, queue, totalRows, width, groupCount, closure.GroupIndices);

                var iteration = 0;
                trace?.RecordFixpointIteration(closure, predicate, iteration, queue.Count, visitedNodes.Count);

                while (queue.Count > 0 && remaining.Count > 0)
                {
                    iteration++;
                    var breadth = queue.Count;
                    for (var i = 0; i < breadth; i++)
                    {
                        var node = queue.Dequeue();
                        ExpandBackwardSearch(seedKey, target, node, remaining, predIndex, maxRequiredIndex, visitedNodes, queue, totalRows, width, groupCount, closure.GroupIndices);
                        if (remaining.Count == 0)
                        {
                            break;
                        }
                    }

                    trace?.RecordFixpointIteration(closure, predicate, iteration, queue.Count, visitedNodes.Count);
                }
            }

            return totalRows;

            void SeedBackwardSearch(
                object[] key,
                object? seed,
                HashSet<object?> remaining,
                Dictionary<RowWrapper, List<object[]>> index,
                int requiredEdgeWidth,
                HashSet<object?> visited,
                Queue<object?> frontier,
                List<object[]> output,
                int rowWidth,
                int groupWidth,
                IReadOnlyList<int> groupIndices)
            {
                visited.Add(seed);
                ExpandBackwardSearch(key, seed, seed, remaining, index, requiredEdgeWidth, visited, frontier, output, rowWidth, groupWidth, groupIndices);
            }

            void ExpandBackwardSearch(
                object[] key,
                object? seed,
                object? current,
                HashSet<object?> remaining,
                Dictionary<RowWrapper, List<object[]>> index,
                int requiredEdgeWidth,
                HashSet<object?> visited,
                Queue<object?> frontier,
                List<object[]> output,
                int rowWidth,
                int groupWidth,
                IReadOnlyList<int> groupIndices)
            {
                var lookup = new object[groupWidth + 1];
                Array.Copy(key, 0, lookup, 0, groupWidth);
                lookup[groupWidth] = current!;

                if (!index.TryGetValue(new RowWrapper(lookup), out var bucket))
                {
                    return;
                }

                foreach (var edge in bucket)
                {
                    if (edge is null || edge.Length <= requiredEdgeWidth)
                    {
                        continue;
                    }

                    var prev = edge[0];
                    if (remaining.Remove(prev))
                    {
                        var rowKey = new object[groupWidth + 2];
                        Array.Copy(key, 0, rowKey, 0, groupWidth);
                        rowKey[groupWidth] = prev!;
                        rowKey[groupWidth + 1] = seed!;
                        output.Add(BuildGroupedClosureRow(rowWidth, groupWidth, groupIndices, rowKey));
                        if (remaining.Count == 0)
                        {
                            return;
                        }
                    }

                    if (!visited.Add(prev))
                    {
                        continue;
                    }

                    frontier.Enqueue(prev);
                }
            }
        }

        private static object[] BuildFlatSeedCacheKey(IReadOnlyList<object[]> seedKeys, int keyWidth)
        {
            if (seedKeys is null) throw new ArgumentNullException(nameof(seedKeys));
            if (keyWidth < 0) throw new ArgumentOutOfRangeException(nameof(keyWidth));

            var key = new object[seedKeys.Count * keyWidth];
            var offset = 0;
            for (var i = 0; i < seedKeys.Count; i++)
            {
                Array.Copy(seedKeys[i], 0, key, offset, keyWidth);
                offset += keyWidth;
            }

            return key;
        }

        private static int CompareCacheSeedRows(object[] left, object[] right)
        {
            if (ReferenceEquals(left, right))
            {
                return 0;
            }

            if (left is null)
            {
                return -1;
            }

            if (right is null)
            {
                return 1;
            }

            var max = Math.Min(left.Length, right.Length);
            for (var i = 0; i < max; i++)
            {
                var comparison = CompareCacheSeedValues(left[i], right[i]);
                if (comparison != 0)
                {
                    return comparison;
                }
            }

            return left.Length.CompareTo(right.Length);
        }

        private static int CompareCacheSeedValues(object? left, object? right)
        {
            if (ReferenceEquals(left, right))
            {
                return 0;
            }

            if (left is null)
            {
                return -1;
            }

            if (right is null)
            {
                return 1;
            }

            if (left is string leftString && right is string rightString)
            {
                return StringComparer.Ordinal.Compare(leftString, rightString);
            }

            var leftType = left.GetType();
            var rightType = right.GetType();

            if (leftType == rightType)
            {
                if (left is IComparable comparable)
                {
                    try
                    {
                        return comparable.CompareTo(right);
                    }
                    catch
                    {
                    }
                }

                return StringComparer.Ordinal.Compare(FormatCacheSeedValue(left), FormatCacheSeedValue(right));
            }

            var leftTypeName = leftType.FullName ?? leftType.Name;
            var rightTypeName = rightType.FullName ?? rightType.Name;
            var typeComparison = StringComparer.Ordinal.Compare(leftTypeName, rightTypeName);
            if (typeComparison != 0)
            {
                return typeComparison;
            }

            return StringComparer.Ordinal.Compare(FormatCacheSeedValue(left), FormatCacheSeedValue(right));
        }

        private static string FormatCacheSeedValue(object value) =>
            value switch
            {
                JsonElement element => element.GetRawText(),
                IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture) ?? string.Empty,
                _ => value.ToString() ?? string.Empty
            };

        private static bool EdgeMatchesFilters(object[] edge, IReadOnlyList<int> groupIndices, object?[] filters)
        {
            for (var i = 0; i < groupIndices.Count; i++)
            {
                var filter = filters[i];
                if (ReferenceEquals(filter, Wildcard.Value))
                {
                    continue;
                }

                if (!Equals(edge[groupIndices[i]], filter))
                {
                    return false;
                }
            }

            return true;
        }

        private static bool TryGetParameterValue(
            object[] tuple,
            IReadOnlyList<int> inputPositions,
            int position,
            out object? value)
        {
            if (tuple.Length == inputPositions.Count)
            {
                for (var i = 0; i < inputPositions.Count; i++)
                {
                    if (inputPositions[i] == position)
                    {
                        value = tuple[i];
                        return true;
                    }
                }

                value = null;
                return false;
            }

            if (position >= 0 && position < tuple.Length)
            {
                value = tuple[position];
                return true;
            }

            value = null;
            return false;
        }

        private IEnumerable<object[]> ExecuteSeededTransitiveClosure(
            TransitiveClosureNode closure,
            IReadOnlyList<object[]> parameters,
            EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));
            if (parameters is null) throw new ArgumentNullException(nameof(parameters));

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                trace?.RecordStrategy(closure, "TransitiveClosureSeeded");

                var predicate = closure.Predicate;
                var edges = GetFactsList(closure.EdgeRelation, context);
                var succIndex = GetFactIndex(closure.EdgeRelation, 0, edges, context);

                var seeds = new List<object?>();
                var seenSeeds = new HashSet<object?>();

                foreach (var paramTuple in parameters)
                {
                    if (paramTuple is null || paramTuple.Length == 0)
                    {
                        continue;
                    }

                    var seed = paramTuple[0];
                    if (seenSeeds.Add(seed))
                    {
                        seeds.Add(seed);
                    }
                }

                seeds.Sort(CompareCacheSeedValues);
                var seedsKey = seeds.ToArray();
                var cacheKey = (closure.EdgeRelation, closure.Predicate);
                var traceKey = $"{closure.Predicate.Name}/{closure.Predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:seeds={seedsKey.Length}";

                if (context.TransitiveClosureSeededResults.TryGetValue(cacheKey, out var cachedBySeed) &&
                    cachedBySeed.TryGetValue(new RowWrapper(seedsKey), out var cachedRows))
                {
                    trace?.RecordCacheLookup("TransitiveClosureSeeded", traceKey, hit: true, built: false);
                    return cachedRows;
                }

                trace?.RecordCacheLookup("TransitiveClosureSeeded", traceKey, hit: false, built: true);

                const int MaxMemoizedSeedCount = 32;
                var canMemoizeSeeds =
                    _cacheContext is not null &&
                    seeds.Count > 1 &&
                    seeds.Count <= MaxMemoizedSeedCount;

                if (canMemoizeSeeds)
                {
                    trace?.RecordStrategy(closure, "TransitiveClosureSeededMemoizedMulti");
                    var memoizedRows = new List<object[]>();

                    foreach (var seed in seeds)
                    {
                        var seedParams = new List<object[]>(1) { new object[] { seed! } };
                        foreach (var row in ExecuteSeededTransitiveClosure(closure, seedParams, context))
                        {
                            memoizedRows.Add(row);
                        }
                    }

                    if (!context.TransitiveClosureSeededResults.TryGetValue(cacheKey, out var memoizedStoreBySeed))
                    {
                        memoizedStoreBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                        context.TransitiveClosureSeededResults.Add(cacheKey, memoizedStoreBySeed);
                    }

                    memoizedStoreBySeed[new RowWrapper(seedsKey)] = memoizedRows;
                    return memoizedRows;
                }

                if (seeds.Count == 1)
                {
                    trace?.RecordStrategy(closure, "TransitiveClosureSeededSingle");

                    var seed = seeds[0];
                    var visitedNodes = new HashSet<object?>();
                    var singleRows = new List<object[]>();
                    var deltaNodes = new List<object?>();

                    var lookupKey = seed ?? NullFactIndexKey;
                    if (succIndex.TryGetValue(lookupKey, out var bucket))
                    {
                        foreach (var edge in bucket)
                        {
                            if (edge is null || edge.Length < 2)
                            {
                                continue;
                            }

                            var to = edge[1];
                            if (visitedNodes.Add(to))
                            {
                                singleRows.Add(new object[] { seed!, to });
                                deltaNodes.Add(to);
                            }
                        }
                    }

                    var singleIteration = 0;
                    trace?.RecordFixpointIteration(closure, predicate, singleIteration, deltaNodes.Count, singleRows.Count);

                    while (deltaNodes.Count > 0)
                    {
                        singleIteration++;
                        var nextDeltaNodes = new List<object?>();

                        foreach (var node in deltaNodes)
                        {
                            lookupKey = node ?? NullFactIndexKey;
                            if (!succIndex.TryGetValue(lookupKey, out bucket))
                            {
                                continue;
                            }

                            foreach (var edge in bucket)
                            {
                                if (edge is null || edge.Length < 2)
                                {
                                    continue;
                                }

                                var next = edge[1];
                                if (visitedNodes.Add(next))
                                {
                                    singleRows.Add(new object[] { seed!, next });
                                    nextDeltaNodes.Add(next);
                                }
                            }
                        }

                        deltaNodes = nextDeltaNodes;
                        trace?.RecordFixpointIteration(closure, predicate, singleIteration, deltaNodes.Count, singleRows.Count);
                    }

                    if (!context.TransitiveClosureSeededResults.TryGetValue(cacheKey, out var singleStoreBySeed))
                    {
                        singleStoreBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                        context.TransitiveClosureSeededResults.Add(cacheKey, singleStoreBySeed);
                    }

                    singleStoreBySeed[new RowWrapper(seedsKey)] = singleRows;
                    return singleRows;
                }

                var visited = new HashSet<PairKey>();
                var totalRows = new List<object[]>();
                var delta = new List<PairKey>();

                foreach (var seed in seeds)
                {
                    var lookupKey = seed ?? NullFactIndexKey;
                    if (!succIndex.TryGetValue(lookupKey, out var bucket))
                    {
                        continue;
                    }

                    foreach (var edge in bucket)
                    {
                        if (edge is null || edge.Length < 2)
                        {
                            continue;
                        }

                        var to = edge[1];
                        var key = new PairKey(seed, to);
                        if (visited.Add(key))
                        {
                            totalRows.Add(new object[] { seed!, to });
                            delta.Add(key);
                        }
                    }
                }

                var iteration = 0;
                trace?.RecordFixpointIteration(closure, predicate, iteration, delta.Count, totalRows.Count);

                while (delta.Count > 0)
                {
                    iteration++;
                    var nextDelta = new List<PairKey>();

                    foreach (var pair in delta)
                    {
                        var lookupKey = pair.To ?? NullFactIndexKey;
                        if (!succIndex.TryGetValue(lookupKey, out var bucket))
                        {
                            continue;
                        }

                        foreach (var edge in bucket)
                        {
                            if (edge is null || edge.Length < 2)
                            {
                                continue;
                            }

                            var next = edge[1];
                            var nextKey = new PairKey(pair.From, next);
                            if (visited.Add(nextKey))
                            {
                                totalRows.Add(new object[] { pair.From!, next });
                                nextDelta.Add(nextKey);
                            }
                        }
                    }

                    delta = nextDelta;
                    trace?.RecordFixpointIteration(closure, predicate, iteration, delta.Count, totalRows.Count);
                }

                if (!context.TransitiveClosureSeededResults.TryGetValue(cacheKey, out var storeBySeed))
                {
                    storeBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                    context.TransitiveClosureSeededResults.Add(cacheKey, storeBySeed);
                }

                storeBySeed[new RowWrapper(seedsKey)] = totalRows;
                return totalRows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private IEnumerable<object[]> ExecuteSeededTransitiveClosureByTarget(
            TransitiveClosureNode closure,
            IReadOnlyList<object[]> parameters,
            EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));
            if (parameters is null) throw new ArgumentNullException(nameof(parameters));

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                trace?.RecordStrategy(closure, "TransitiveClosureSeededByTarget");

                var predicate = closure.Predicate;
                var edges = GetFactsList(closure.EdgeRelation, context);
                var predIndex = GetFactIndex(closure.EdgeRelation, 1, edges, context);

                var seeds = new List<object?>();
                var seenSeeds = new HashSet<object?>();

                foreach (var paramTuple in parameters)
                {
                    if (paramTuple is null || paramTuple.Length == 0)
                    {
                        continue;
                    }

                    object? seed = null;
                    if (paramTuple.Length == 1)
                    {
                        seed = paramTuple[0];
                    }
                    else if (paramTuple.Length > 1)
                    {
                        seed = paramTuple[1];
                    }

                    if (seenSeeds.Add(seed))
                    {
                        seeds.Add(seed);
                    }
                }

                seeds.Sort(CompareCacheSeedValues);
                var seedsKey = seeds.ToArray();
                var cacheKey = (closure.EdgeRelation, closure.Predicate);
                var traceKey = $"{closure.Predicate.Name}/{closure.Predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:targets={seedsKey.Length}";

                if (context.TransitiveClosureSeededByTargetResults.TryGetValue(cacheKey, out var cachedBySeed) &&
                    cachedBySeed.TryGetValue(new RowWrapper(seedsKey), out var cachedRows))
                {
                    trace?.RecordCacheLookup("TransitiveClosureSeededByTarget", traceKey, hit: true, built: false);
                    return cachedRows;
                }

                trace?.RecordCacheLookup("TransitiveClosureSeededByTarget", traceKey, hit: false, built: true);

                const int MaxMemoizedTargetSeedCount = 32;
                var canMemoizeSeeds =
                    _cacheContext is not null &&
                    seeds.Count > 1 &&
                    seeds.Count <= MaxMemoizedTargetSeedCount;

                if (canMemoizeSeeds)
                {
                    trace?.RecordStrategy(closure, "TransitiveClosureSeededByTargetMemoizedMulti");
                    var memoizedRows = new List<object[]>();

                    foreach (var seed in seeds)
                    {
                        var seedParams = new List<object[]>(1) { new object[] { seed! } };
                        foreach (var row in ExecuteSeededTransitiveClosureByTarget(closure, seedParams, context))
                        {
                            memoizedRows.Add(row);
                        }
                    }

                    if (!context.TransitiveClosureSeededByTargetResults.TryGetValue(cacheKey, out var memoizedStoreBySeed))
                    {
                        memoizedStoreBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                        context.TransitiveClosureSeededByTargetResults.Add(cacheKey, memoizedStoreBySeed);
                    }

                    memoizedStoreBySeed[new RowWrapper(seedsKey)] = memoizedRows;
                    return memoizedRows;
                }

                if (seeds.Count == 1)
                {
                    trace?.RecordStrategy(closure, "TransitiveClosureSeededByTargetSingle");

                    var seed = seeds[0];
                    var visitedNodes = new HashSet<object?>();
                    var singleRows = new List<object[]>();
                    var deltaNodes = new List<object?>();

                    var lookupKey = seed ?? NullFactIndexKey;
                    if (predIndex.TryGetValue(lookupKey, out var bucket))
                    {
                        foreach (var edge in bucket)
                        {
                            if (edge is null || edge.Length < 2)
                            {
                                continue;
                            }

                            var from = edge[0];
                            if (visitedNodes.Add(from))
                            {
                                singleRows.Add(new object[] { from!, seed! });
                                deltaNodes.Add(from);
                            }
                        }
                    }

                    var singleIteration = 0;
                    trace?.RecordFixpointIteration(closure, predicate, singleIteration, deltaNodes.Count, singleRows.Count);

                    while (deltaNodes.Count > 0)
                    {
                        singleIteration++;
                        var nextDeltaNodes = new List<object?>();

                        foreach (var node in deltaNodes)
                        {
                            lookupKey = node ?? NullFactIndexKey;
                            if (!predIndex.TryGetValue(lookupKey, out bucket))
                            {
                                continue;
                            }

                            foreach (var edge in bucket)
                            {
                                if (edge is null || edge.Length < 2)
                                {
                                    continue;
                                }

                                var prev = edge[0];
                                if (visitedNodes.Add(prev))
                                {
                                    singleRows.Add(new object[] { prev!, seed! });
                                    nextDeltaNodes.Add(prev);
                                }
                            }
                        }

                        deltaNodes = nextDeltaNodes;
                        trace?.RecordFixpointIteration(closure, predicate, singleIteration, deltaNodes.Count, singleRows.Count);
                    }

                    if (!context.TransitiveClosureSeededByTargetResults.TryGetValue(cacheKey, out var singleStoreBySeed))
                    {
                        singleStoreBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                        context.TransitiveClosureSeededByTargetResults.Add(cacheKey, singleStoreBySeed);
                    }

                    singleStoreBySeed[new RowWrapper(seedsKey)] = singleRows;
                    return singleRows;
                }

                var visited = new HashSet<PairKey>();
                var totalRows = new List<object[]>();
                var delta = new List<PairKey>();

                foreach (var seed in seeds)
                {
                    var lookupKey = seed ?? NullFactIndexKey;
                    if (!predIndex.TryGetValue(lookupKey, out var bucket))
                    {
                        continue;
                    }

                    foreach (var edge in bucket)
                    {
                        if (edge is null || edge.Length < 2)
                        {
                            continue;
                        }

                        var from = edge[0];
                        var key = new PairKey(from, seed);
                        if (visited.Add(key))
                        {
                            totalRows.Add(new object[] { from, seed! });
                            delta.Add(key);
                        }
                    }
                }

                var iteration = 0;
                trace?.RecordFixpointIteration(closure, predicate, iteration, delta.Count, totalRows.Count);

                while (delta.Count > 0)
                {
                    iteration++;
                    var nextDelta = new List<PairKey>();

                    foreach (var pair in delta)
                    {
                        var lookupKey = pair.From ?? NullFactIndexKey;
                        if (!predIndex.TryGetValue(lookupKey, out var bucket))
                        {
                            continue;
                        }

                        foreach (var edge in bucket)
                        {
                            if (edge is null || edge.Length < 2)
                            {
                                continue;
                            }

                            var prev = edge[0];
                            var nextKey = new PairKey(prev, pair.To);
                            if (visited.Add(nextKey))
                            {
                                totalRows.Add(new object[] { prev, pair.To! });
                                nextDelta.Add(nextKey);
                            }
                        }
                    }

                    delta = nextDelta;
                    trace?.RecordFixpointIteration(closure, predicate, iteration, delta.Count, totalRows.Count);
                }

                if (!context.TransitiveClosureSeededByTargetResults.TryGetValue(cacheKey, out var storeBySeed))
                {
                    storeBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                    context.TransitiveClosureSeededByTargetResults.Add(cacheKey, storeBySeed);
                }

                storeBySeed[new RowWrapper(seedsKey)] = totalRows;
                return totalRows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private IEnumerable<object[]> ExecuteSeededTransitiveClosurePairs(
            TransitiveClosureNode closure,
            IReadOnlyList<object[]> parameters,
            EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));
            if (parameters is null) throw new ArgumentNullException(nameof(parameters));

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;

                var bySource = new Dictionary<object?, HashSet<object?>>();
                var byTarget = new Dictionary<object?, HashSet<object?>>();

                foreach (var paramTuple in parameters)
                {
                    if (paramTuple is null || paramTuple.Length == 0)
                    {
                        continue;
                    }

                    object? source = null;
                    object? target = null;

                    if (paramTuple.Length >= 2)
                    {
                        source = paramTuple[0];
                        target = paramTuple[1];
                    }
                    else if (paramTuple.Length == 1)
                    {
                        source = paramTuple[0];
                        target = null;
                    }

                    if (!bySource.TryGetValue(source, out var targets))
                    {
                        targets = new HashSet<object?>();
                        bySource.Add(source, targets);
                    }
                    targets.Add(target);

                    if (!byTarget.TryGetValue(target, out var sources))
                    {
                        sources = new HashSet<object?>();
                        byTarget.Add(target, sources);
                    }
                    sources.Add(source);
                }

                trace?.RecordStrategy(closure, "TransitiveClosurePairs");

                const int MaxMemoizedPairSeeds = 32;
                var canMemoizePairs =
                    _cacheContext is not null &&
                    bySource.Count <= MaxMemoizedPairSeeds &&
                    byTarget.Count <= MaxMemoizedPairSeeds;

                if (canMemoizePairs && bySource.Count <= byTarget.Count)
                {
                    trace?.RecordStrategy(closure, "TransitiveClosurePairsMemoized");
                    trace?.RecordStrategy(closure, "TransitiveClosurePairsMemoizedForward");
                    return ExecuteSeededTransitiveClosurePairsMemoizedBySource(closure, bySource, context);
                }

                if (canMemoizePairs)
                {
                    trace?.RecordStrategy(closure, "TransitiveClosurePairsMemoized");
                    trace?.RecordStrategy(closure, "TransitiveClosurePairsMemoizedBackward");
                    return ExecuteSeededTransitiveClosurePairsMemoizedByTarget(closure, byTarget, context);
                }

                if (bySource.Count <= byTarget.Count)
                {
                    trace?.RecordStrategy(closure, "TransitiveClosurePairsForward");
                    return ExecuteSeededTransitiveClosurePairsForward(closure, bySource, context);
                }

                trace?.RecordStrategy(closure, "TransitiveClosurePairsBackward");
                return ExecuteSeededTransitiveClosurePairsBackward(closure, byTarget, context);
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private IEnumerable<object[]> ExecuteSeededTransitiveClosurePairsMemoizedBySource(
            TransitiveClosureNode closure,
            IReadOnlyDictionary<object?, HashSet<object?>> targetsBySource,
            EvaluationContext context)
        {
            var totalRows = new List<object[]>();

            foreach (var entry in targetsBySource)
            {
                var source = entry.Key;
                var targets = entry.Value;
                if (targets.Count == 0)
                {
                    continue;
                }

                var seedParams = new List<object[]>(1) { new object[] { source! } };
                var reachableRows = ExecuteSeededTransitiveClosure(closure, seedParams, context);

                if (targets.Contains(null))
                {
                    foreach (var row in reachableRows)
                    {
                        totalRows.Add(row);
                    }

                    continue;
                }

                foreach (var row in reachableRows)
                {
                    if (row is null || row.Length < 2)
                    {
                        continue;
                    }

                    if (targets.Contains(row[1]))
                    {
                        totalRows.Add(row);
                    }
                }
            }

            return totalRows;
        }

        private IEnumerable<object[]> ExecuteSeededTransitiveClosurePairsMemoizedByTarget(
            TransitiveClosureNode closure,
            IReadOnlyDictionary<object?, HashSet<object?>> sourcesByTarget,
            EvaluationContext context)
        {
            var totalRows = new List<object[]>();

            foreach (var entry in sourcesByTarget)
            {
                var target = entry.Key;
                var sources = entry.Value;
                if (sources.Count == 0)
                {
                    continue;
                }

                var seedParams = new List<object[]>(1) { new object[] { target! } };
                var reachableRows = ExecuteSeededTransitiveClosureByTarget(closure, seedParams, context);

                if (sources.Contains(null))
                {
                    foreach (var row in reachableRows)
                    {
                        totalRows.Add(row);
                    }

                    continue;
                }

                foreach (var row in reachableRows)
                {
                    if (row is null || row.Length < 2)
                    {
                        continue;
                    }

                    if (sources.Contains(row[0]))
                    {
                        totalRows.Add(row);
                    }
                }
            }

            return totalRows;
        }

        private IEnumerable<object[]> ExecuteSeededTransitiveClosurePairsForward(
            TransitiveClosureNode closure,
            IReadOnlyDictionary<object?, HashSet<object?>> targetsBySource,
            EvaluationContext context)
        {
            var trace = context.Trace;
            var predicate = closure.Predicate;
            var edges = GetFactsList(closure.EdgeRelation, context);
            var succIndex = GetFactIndex(closure.EdgeRelation, 0, edges, context);

            var totalRows = new List<object[]>();

            foreach (var entry in targetsBySource)
            {
                var source = entry.Key;
                if (entry.Value.Count == 0)
                {
                    continue;
                }

                var remainingTargets = new HashSet<object?>(entry.Value);
                var visitedNodes = new HashSet<object?>();
                var queue = new Queue<object?>();

                SeedForwardSearch(source, remainingTargets, succIndex, visitedNodes, queue, totalRows);

                var iteration = 0;
                var deltaCount = queue.Count;
                trace?.RecordFixpointIteration(closure, predicate, iteration, deltaCount, visitedNodes.Count);

                while (queue.Count > 0 && remainingTargets.Count > 0)
                {
                    iteration++;
                    var breadth = queue.Count;
                    for (var i = 0; i < breadth; i++)
                    {
                        var node = queue.Dequeue();
                        ExpandForwardSearch(source, node, remainingTargets, succIndex, visitedNodes, queue, totalRows);
                        if (remainingTargets.Count == 0)
                        {
                            break;
                        }
                    }

                    trace?.RecordFixpointIteration(closure, predicate, iteration, queue.Count, visitedNodes.Count);
                }
            }

            return totalRows;

            void SeedForwardSearch(
                object? seed,
                HashSet<object?> remaining,
                Dictionary<object, List<object[]>> index,
                HashSet<object?> visited,
                Queue<object?> frontier,
                List<object[]> output)
            {
                var lookupKey = seed ?? NullFactIndexKey;
                if (!index.TryGetValue(lookupKey, out var bucket))
                {
                    return;
                }

                foreach (var edge in bucket)
                {
                    if (edge is null || edge.Length < 2)
                    {
                        continue;
                    }

                    var next = edge[1];
                    if (!visited.Add(next))
                    {
                        continue;
                    }

                    frontier.Enqueue(next);
                    if (remaining.Remove(next))
                    {
                        output.Add(new object[] { seed!, next });
                        if (remaining.Count == 0)
                        {
                            return;
                        }
                    }
                }
            }

            void ExpandForwardSearch(
                object? seed,
                object? current,
                HashSet<object?> remaining,
                Dictionary<object, List<object[]>> index,
                HashSet<object?> visited,
                Queue<object?> frontier,
                List<object[]> output)
            {
                var lookupKey = current ?? NullFactIndexKey;
                if (!index.TryGetValue(lookupKey, out var bucket))
                {
                    return;
                }

                foreach (var edge in bucket)
                {
                    if (edge is null || edge.Length < 2)
                    {
                        continue;
                    }

                    var next = edge[1];
                    if (!visited.Add(next))
                    {
                        continue;
                    }

                    frontier.Enqueue(next);
                    if (remaining.Remove(next))
                    {
                        output.Add(new object[] { seed!, next });
                        if (remaining.Count == 0)
                        {
                            return;
                        }
                    }
                }
            }
        }

        private IEnumerable<object[]> ExecuteSeededTransitiveClosurePairsBackward(
            TransitiveClosureNode closure,
            IReadOnlyDictionary<object?, HashSet<object?>> sourcesByTarget,
            EvaluationContext context)
        {
            var trace = context.Trace;
            var predicate = closure.Predicate;
            var edges = GetFactsList(closure.EdgeRelation, context);
            var predIndex = GetFactIndex(closure.EdgeRelation, 1, edges, context);

            var totalRows = new List<object[]>();

            foreach (var entry in sourcesByTarget)
            {
                var target = entry.Key;
                if (entry.Value.Count == 0)
                {
                    continue;
                }

                var remainingSources = new HashSet<object?>(entry.Value);
                var visitedNodes = new HashSet<object?>();
                var queue = new Queue<object?>();

                SeedBackwardSearch(target, remainingSources, predIndex, visitedNodes, queue, totalRows);

                var iteration = 0;
                var deltaCount = queue.Count;
                trace?.RecordFixpointIteration(closure, predicate, iteration, deltaCount, visitedNodes.Count);

                while (queue.Count > 0 && remainingSources.Count > 0)
                {
                    iteration++;
                    var breadth = queue.Count;
                    for (var i = 0; i < breadth; i++)
                    {
                        var node = queue.Dequeue();
                        ExpandBackwardSearch(target, node, remainingSources, predIndex, visitedNodes, queue, totalRows);
                        if (remainingSources.Count == 0)
                        {
                            break;
                        }
                    }

                    trace?.RecordFixpointIteration(closure, predicate, iteration, queue.Count, visitedNodes.Count);
                }
            }

            return totalRows;

            void SeedBackwardSearch(
                object? seed,
                HashSet<object?> remaining,
                Dictionary<object, List<object[]>> index,
                HashSet<object?> visited,
                Queue<object?> frontier,
                List<object[]> output)
            {
                var lookupKey = seed ?? NullFactIndexKey;
                if (!index.TryGetValue(lookupKey, out var bucket))
                {
                    return;
                }

                foreach (var edge in bucket)
                {
                    if (edge is null || edge.Length < 2)
                    {
                        continue;
                    }

                    var prev = edge[0];
                    if (!visited.Add(prev))
                    {
                        continue;
                    }

                    frontier.Enqueue(prev);
                    if (remaining.Remove(prev))
                    {
                        output.Add(new object[] { prev!, seed });
                        if (remaining.Count == 0)
                        {
                            return;
                        }
                    }
                }
            }

            void ExpandBackwardSearch(
                object? seed,
                object? current,
                HashSet<object?> remaining,
                Dictionary<object, List<object[]>> index,
                HashSet<object?> visited,
                Queue<object?> frontier,
                List<object[]> output)
            {
                var lookupKey = current ?? NullFactIndexKey;
                if (!index.TryGetValue(lookupKey, out var bucket))
                {
                    return;
                }

                foreach (var edge in bucket)
                {
                    if (edge is null || edge.Length < 2)
                    {
                        continue;
                    }

                    var prev = edge[0];
                    if (!visited.Add(prev))
                    {
                        continue;
                    }

                    frontier.Enqueue(prev);
                    if (remaining.Remove(prev))
                    {
                        output.Add(new object[] { prev!, seed });
                        if (remaining.Count == 0)
                        {
                            return;
                        }
                    }
                }
            }
        }

        private IEnumerable<object[]> ExecuteFixpoint(FixpointNode fixpoint, EvaluationContext? parentContext)
        {
            if (fixpoint is null) throw new ArgumentNullException(nameof(fixpoint));

            var comparer = StructuralArrayComparer.Instance;
            var predicate = fixpoint.Predicate;
            var totalSet = new HashSet<RowWrapper>(new RowWrapperComparer(comparer));
            var totalRows = new List<object[]>();
            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                var baseRows = Evaluate(fixpoint.BasePlan, context).ToList();
                var deltaRows = new List<object[]>();

                foreach (var tuple in baseRows)
                {
                    if (TryAddRow(totalSet, tuple))
                    {
                        totalRows.Add(tuple);
                        deltaRows.Add(tuple);
                    }
                }

                context.Current = predicate;
                context.Totals[predicate] = totalRows;
                context.Deltas[predicate] = deltaRows;

                var iteration = 0;
                trace?.RecordFixpointIteration(fixpoint, predicate, iteration, deltaRows.Count, totalRows.Count);

                while (context.Deltas.TryGetValue(predicate, out var delta) && delta.Count > 0)
                {
                    iteration++;
                    var nextDelta = new List<object[]>();
                    foreach (var recursivePlan in fixpoint.RecursivePlans)
                    {
                        foreach (var tuple in Evaluate(recursivePlan, context))
                        {
                            if (TryAddRow(totalSet, tuple))
                            {
                                nextDelta.Add(tuple);
                            }
                        }
                    }
                    totalRows.AddRange(nextDelta);
                    context.Deltas[predicate] = nextDelta;
                    trace?.RecordFixpointIteration(fixpoint, predicate, iteration, nextDelta.Count, totalRows.Count);
                }

                return totalRows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private IEnumerable<object[]> ExecuteMutualFixpoint(MutualFixpointNode node, EvaluationContext? parentContext)
        {
            if (node is null) throw new ArgumentNullException(nameof(node));

            var comparer = StructuralArrayComparer.Instance;
            var totalSets = new Dictionary<PredicateId, HashSet<RowWrapper>>();
            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;

                foreach (var member in node.Members)
                {
                    var predicate = member.Predicate;
                    var totalList = new List<object[]>();
                    var deltaList = new List<object[]>();
                    context.Totals[predicate] = totalList;
                    context.Deltas[predicate] = deltaList;

                    var set = new HashSet<RowWrapper>(new RowWrapperComparer(comparer));
                    totalSets[predicate] = set;

                    var baseRows = Evaluate(member.BasePlan, context).ToList();
                    foreach (var tuple in baseRows)
                    {
                        if (TryAddRow(set, tuple))
                        {
                            totalList.Add(tuple);
                            deltaList.Add(tuple);
                        }
                    }
                }

                var iteration = 0;
                if (trace is not null)
                {
                    foreach (var member in node.Members)
                    {
                        var predicate = member.Predicate;
                        trace.RecordFixpointIteration(node, predicate, iteration, context.Deltas[predicate].Count, context.Totals[predicate].Count);
                    }
                }

                while (context.Deltas.Values.Any(delta => delta.Count > 0))
                {
                    iteration++;
                    var nextDeltas = node.Members.ToDictionary(member => member.Predicate, _ => new List<object[]>());

                    foreach (var member in node.Members)
                    {
                        context.Current = member.Predicate;
                        var predicate = member.Predicate;
                        var totalList = context.Totals[predicate];
                        var totalSet = totalSets[predicate];
                        var memberNext = nextDeltas[predicate];

                        foreach (var recursivePlan in member.RecursivePlans)
                        {
                            foreach (var tuple in Evaluate(recursivePlan, context))
                            {
                                if (TryAddRow(totalSet, tuple))
                                {
                                    memberNext.Add(tuple);
                                }
                            }
                        }

                        totalList.AddRange(memberNext);
                    }

                    foreach (var pair in nextDeltas)
                    {
                        context.Deltas[pair.Key] = pair.Value;
                    }

                    if (trace is not null)
                    {
                        foreach (var member in node.Members)
                        {
                            var predicate = member.Predicate;
                            trace.RecordFixpointIteration(node, predicate, iteration, context.Deltas[predicate].Count, context.Totals[predicate].Count);
                        }
                    }
                }

                return context.Totals.TryGetValue(node.Head, out var headRows)
                    ? headRows
                    : Array.Empty<object[]>();
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private IEnumerable<object[]> EvaluateRecursiveReference(RecursiveRefNode node, EvaluationContext? context)
        {
            if (context is null)
            {
                throw new InvalidOperationException("Recursive reference evaluated without an execution context.");
            }

            if (!node.Predicate.Equals(context.Current))
            {
                throw new NotSupportedException($"Cross-predicate recursion is not supported (referenced {node.Predicate} while evaluating {context.Current}).");
            }

            return ResolveReference(node.Predicate, node.Kind, context);
        }

        private IEnumerable<object[]> EvaluateCrossReference(CrossRefNode node, EvaluationContext? context)
        {
            if (context is null)
            {
                throw new InvalidOperationException("Mutual recursion reference evaluated without an execution context.");
            }

            return ResolveReference(node.Predicate, node.Kind, context);
        }

        private static IEnumerable<object[]> ResolveReference(PredicateId predicate, RecursiveRefKind kind, EvaluationContext context)
        {
            var map = kind switch
            {
                RecursiveRefKind.Total => context.Totals,
                RecursiveRefKind.Delta => context.Deltas,
                _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unknown recursive reference kind.")
            };

            return map.TryGetValue(predicate, out var rows)
                ? rows
                : Array.Empty<object[]>();
        }

        private object EvaluateArithmeticExpression(ArithmeticExpression expression, object[] tuple) =>
            expression switch
            {
                ColumnExpression column => tuple[column.Index],
                ConstantExpression constant => constant.Value,
                UnaryArithmeticExpression unary => ApplyUnary(unary.Operator, EvaluateArithmeticExpression(unary.Operand, tuple)),
                BinaryArithmeticExpression binary => ApplyBinary(binary.Operator,
                    EvaluateArithmeticExpression(binary.Left, tuple),
                    EvaluateArithmeticExpression(binary.Right, tuple)),
                _ => throw new NotSupportedException($"Unsupported arithmetic expression: {expression?.GetType().Name}")
            };

        private object ApplyUnary(ArithmeticUnaryOperator op, object operand) =>
            op switch
            {
                ArithmeticUnaryOperator.Negate => NegateValue(operand),
                _ => throw new ArgumentOutOfRangeException(nameof(op), op, "Unknown arithmetic unary operator.")
            };

        private object ApplyBinary(ArithmeticBinaryOperator op, object left, object right) =>
            op switch
            {
                ArithmeticBinaryOperator.Add => AddValues(left, right),
                ArithmeticBinaryOperator.Subtract => SubtractValues(left, right),
                ArithmeticBinaryOperator.Multiply => MultiplyValues(left, right),
                ArithmeticBinaryOperator.Divide => DivideValues(left, right),
                ArithmeticBinaryOperator.IntegerDivide => IntegerDivideValues(left, right),
                ArithmeticBinaryOperator.Modulo => ModuloValues(left, right),
                _ => throw new ArgumentOutOfRangeException(nameof(op), op, "Unknown arithmetic binary operator.")
            };

        private object NegateValue(object value)
        {
            if (IsIntegral(value))
            {
                var operand = ToInt64(value);
                var result = checked(-operand);
                return CreateIntegralResult(result);
            }

            var numeric = ToDouble(value);
            return -numeric;
        }

        private object AddValues(object left, object right)
        {
            if (IsIntegral(left) && IsIntegral(right))
            {
                var a = ToInt64(left);
                var b = ToInt64(right);
                var result = checked(a + b);
                return CreateIntegralResult(result);
            }

            return ToDouble(left) + ToDouble(right);
        }

        private object SubtractValues(object left, object right)
        {
            if (IsIntegral(left) && IsIntegral(right))
            {
                var a = ToInt64(left);
                var b = ToInt64(right);
                var result = checked(a - b);
                return CreateIntegralResult(result);
            }

            return ToDouble(left) - ToDouble(right);
        }

        private object MultiplyValues(object left, object right)
        {
            if (IsIntegral(left) && IsIntegral(right))
            {
                var a = ToInt64(left);
                var b = ToInt64(right);
                var result = checked(a * b);
                return CreateIntegralResult(result);
            }

            return ToDouble(left) * ToDouble(right);
        }

        private object DivideValues(object left, object right)
        {
            var divisor = ToDouble(right);
            if (Math.Abs(divisor) < double.Epsilon)
            {
                throw new DivideByZeroException("Division by zero in arithmetic expression.");
            }

            return ToDouble(left) / divisor;
        }

        private object IntegerDivideValues(object left, object right)
        {
            if (!(IsIntegral(left) && IsIntegral(right)))
            {
                throw new InvalidOperationException("Integer division requires integral operands.");
            }

            var divisor = ToInt64(right);
            if (divisor == 0)
            {
                throw new DivideByZeroException("Integer division by zero in arithmetic expression.");
            }

            var dividend = ToInt64(left);
            var result = dividend / divisor;
            return CreateIntegralResult(result);
        }

        private object ModuloValues(object left, object right)
        {
            if (!(IsIntegral(left) && IsIntegral(right)))
            {
                throw new InvalidOperationException("Modulo requires integral operands.");
            }

            var divisor = ToInt64(right);
            if (divisor == 0)
            {
                throw new DivideByZeroException("Modulo division by zero in arithmetic expression.");
            }

            var dividend = ToInt64(left);
            var result = dividend % divisor;
            return CreateIntegralResult(result);
        }

        private static bool IsIntegral(object value) =>
            value is sbyte or byte or short or ushort or int or uint or long or ulong;

        private static long ToInt64(object value) => Convert.ToInt64(value);

        private static double ToDouble(object value) => Convert.ToDouble(value);

        private static object CreateIntegralResult(long value) =>
            value is >= int.MinValue and <= int.MaxValue ? (object)(int)value : value;

        public static int CompareValues(object left, object right)
        {
            if (left is IComparable comparableLeft && right is IComparable comparableRight)
            {
                if (left.GetType() == right.GetType())
                {
                    return comparableLeft.CompareTo(right);
                }
            }

            var leftNumeric = Convert.ToDouble(left);
            var rightNumeric = Convert.ToDouble(right);
            return leftNumeric.CompareTo(rightNumeric);
        }

        private static bool TryAddRow(HashSet<RowWrapper> set, object[] tuple)
        {
            if (tuple is null) throw new ArgumentNullException(nameof(tuple));
            return set.Add(new RowWrapper(tuple));
        }

        private sealed class EvaluationContext
        {
            public EvaluationContext(
                IEnumerable<object[]>? parameters = null,
                EvaluationContext? parent = null,
                QueryExecutionTrace? trace = null,
                CancellationToken cancellationToken = default)
            {
                Parameters = parameters?.ToList() ?? new List<object[]>();
                Trace = trace ?? parent?.Trace;
                CancellationToken = cancellationToken.CanBeCanceled ? cancellationToken : parent?.CancellationToken ?? cancellationToken;
                FixpointDepth = parent?.FixpointDepth ?? 0;
                Facts = parent?.Facts ?? new Dictionary<PredicateId, List<object[]>>();
                FactSources = parent?.FactSources ?? new Dictionary<PredicateId, PlanNode>();
                FactSets = parent?.FactSets ?? new Dictionary<PredicateId, HashSet<object[]>>();
                FactIndices = parent?.FactIndices ?? new Dictionary<(PredicateId Predicate, int ColumnIndex), Dictionary<object, List<object[]>>>();
                JoinIndices = parent?.JoinIndices ?? new Dictionary<(PredicateId Predicate, string KeySignature), Dictionary<RowWrapper, List<object[]>>>();
                TransitiveClosureResults = parent?.TransitiveClosureResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId Predicate), IReadOnlyList<object[]>>();
                TransitiveClosureSeededResults = parent?.TransitiveClosureSeededResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId Predicate), Dictionary<RowWrapper, IReadOnlyList<object[]>>>();
                TransitiveClosureSeededByTargetResults = parent?.TransitiveClosureSeededByTargetResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId Predicate), Dictionary<RowWrapper, IReadOnlyList<object[]>>>();
                GroupedTransitiveClosureResults = parent?.GroupedTransitiveClosureResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId Predicate, string Groups), IReadOnlyList<object[]>>();
                GroupedTransitiveClosureSeededResults = parent?.GroupedTransitiveClosureSeededResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId Predicate, string Groups), Dictionary<RowWrapper, IReadOnlyList<object[]>>>();
                GroupedTransitiveClosureSeededByTargetResults = parent?.GroupedTransitiveClosureSeededByTargetResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId Predicate, string Groups), Dictionary<RowWrapper, IReadOnlyList<object[]>>>();
            }

            public PredicateId Current { get; set; }
            = new PredicateId("", 0);

            public Dictionary<PredicateId, List<object[]>> Totals { get; } = new();

            public Dictionary<PredicateId, List<object[]>> Deltas { get; } = new();

            public IReadOnlyList<object[]> Parameters { get; }

            public QueryExecutionTrace? Trace { get; }

            public CancellationToken CancellationToken { get; }

            public int FixpointDepth { get; set; }

            public Dictionary<string, List<object[]>> Materialized { get; } = new();

            public Dictionary<(string Id, int ColumnIndex), Dictionary<object, List<object[]>>> MaterializeFactIndices { get; } = new();

            public Dictionary<(string Id, string KeySignature), Dictionary<RowWrapper, List<object[]>>> MaterializeJoinIndices { get; } = new();

            public Dictionary<(PredicateId Predicate, RecursiveRefKind Kind, int ColumnIndex), IncrementalFactIndexCache> RecursiveFactIndices { get; } =
                new();

            public Dictionary<(PredicateId Predicate, RecursiveRefKind Kind, string KeySignature), IncrementalJoinIndexCache> RecursiveJoinIndices { get; } =
                new();

            public Dictionary<PredicateId, List<object[]>> Facts { get; }

            public Dictionary<PredicateId, PlanNode> FactSources { get; }

            public Dictionary<PredicateId, HashSet<object[]>> FactSets { get; }

            public Dictionary<(PredicateId Predicate, int ColumnIndex), Dictionary<object, List<object[]>>> FactIndices { get; }

            public Dictionary<(PredicateId Predicate, string KeySignature), Dictionary<RowWrapper, List<object[]>>> JoinIndices { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId Predicate), IReadOnlyList<object[]>> TransitiveClosureResults { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId Predicate), Dictionary<RowWrapper, IReadOnlyList<object[]>>> TransitiveClosureSeededResults { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId Predicate), Dictionary<RowWrapper, IReadOnlyList<object[]>>> TransitiveClosureSeededByTargetResults { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId Predicate, string Groups), IReadOnlyList<object[]>> GroupedTransitiveClosureResults { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId Predicate, string Groups), Dictionary<RowWrapper, IReadOnlyList<object[]>>> GroupedTransitiveClosureSeededResults { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId Predicate, string Groups), Dictionary<RowWrapper, IReadOnlyList<object[]>>> GroupedTransitiveClosureSeededByTargetResults { get; }
        }

        private readonly record struct PairKey(object? From, object? To);

        private readonly struct RowWrapper
        {
            public RowWrapper(object[] row)
            {
                Row = row ?? throw new ArgumentNullException(nameof(row));
            }

            public object[] Row { get; }
        }

        private sealed class RowWrapperComparer : IEqualityComparer<RowWrapper>
        {
            private readonly IEqualityComparer<object[]> _inner;

            public RowWrapperComparer(IEqualityComparer<object[]> inner)
            {
                _inner = inner;
            }

            public bool Equals(RowWrapper x, RowWrapper y) => _inner.Equals(x.Row, y.Row);

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

        private static bool TryGetArrayElement(JsonElement arrayElement, int index, out JsonElement value)
        {
            if (index < 0)
            {
                value = default;
                return false;
            }
            var currentIndex = 0;
            foreach (var child in arrayElement.EnumerateArray())
            {
                if (currentIndex == index)
                {
                    value = child;
                    return true;
                }
                currentIndex++;
            }
            value = default;
            return false;
        }
    }
}

namespace UnifyWeaver.QueryRuntime.Dynamic
{
    public enum RecordSeparatorKind
    {
        LineFeed,
        Null
    }

    public enum QuoteStyle
    {
        None,
        DoubleQuote,
        SingleQuote,
        Json
    }

    public sealed record DynamicSourceConfig
    {
        public string InputPath { get; init; } = string.Empty;
        public string FieldSeparator { get; init; } = ",";
        public RecordSeparatorKind RecordSeparator { get; init; } = RecordSeparatorKind.LineFeed;
        public QuoteStyle QuoteStyle { get; init; } = QuoteStyle.None;
        public int SkipRows { get; init; } = 0;
        public int ExpectedWidth { get; init; } = 0;
    }

    public static class XmlDefaults
    {
        public static readonly IReadOnlyDictionary<string, string> BuiltinNamespacePrefixes =
            new Dictionary<string, string>(StringComparer.Ordinal)
            {
                { "http://www.pearltrees.com/rdf/0.1/#", "pt" },
                { "http://purl.org/dc/elements/1.1/", "dcterms" }
            };
    }

    public sealed record XmlSourceConfig
    {
        public string InputPath { get; init; } = string.Empty;
        public Stream? InputStream { get; init; } = null;
        public RecordSeparatorKind RecordSeparator { get; init; } = RecordSeparatorKind.Null;
        public int ExpectedWidth { get; init; } = 1;
        public IReadOnlyDictionary<string, string>? NamespacePrefixes { get; init; } = XmlDefaults.BuiltinNamespacePrefixes;
        public bool TreatPearltreesCDataAsText { get; init; } = true;
        public bool NestedProjection { get; init; } = false;
    }

    public sealed class DelimitedTextReader
    {
        private readonly DynamicSourceConfig _config;

        public DelimitedTextReader(DynamicSourceConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            if (string.IsNullOrWhiteSpace(_config.InputPath))
                throw new ArgumentException("InputPath is required", nameof(config));
            if (string.IsNullOrEmpty(_config.FieldSeparator))
                throw new ArgumentException("FieldSeparator is required", nameof(config));
            if (_config.ExpectedWidth <= 0)
                throw new ArgumentException("ExpectedWidth must be positive", nameof(config));
        }

        public IEnumerable<object[]> Read()
        {
            return _config.RecordSeparator == RecordSeparatorKind.Null
                ? ReadNullSeparated()
                : ReadLineSeparated();
        }

        private IEnumerable<object[]> ReadLineSeparated()
        {
            using var reader = OpenReader();
            SkipRows(reader);
            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                if (line.Length == 0)
                {
                    continue;
                }
                yield return ParseRecord(line);
            }
        }

        private IEnumerable<object[]> ReadNullSeparated()
        {
            using var reader = OpenReader();
            var text = reader.ReadToEnd();
            var parts = text.Split('\0');
            for (int i = _config.SkipRows; i < parts.Length; i++)
            {
                var record = parts[i];
                if (record.Length == 0)
                {
                    continue;
                }
                yield return ParseRecord(record);
            }
        }

        private void SkipRows(StreamReader reader)
        {
            for (int i = 0; i < _config.SkipRows; i++)
            {
                if (reader.ReadLine() is null)
                {
                    break;
                }
            }
        }

        private object[] ParseRecord(string record)
        {
            List<string> fields = _config.QuoteStyle == QuoteStyle.None
                ? ParseWithoutQuotes(record)
                : ParseWithQuotes(record);

            if (fields.Count != _config.ExpectedWidth)
            {
                throw new InvalidDataException($"Expected {_config.ExpectedWidth} columns but found {fields.Count} in record: {record}");
            }
            return fields.Select(f => (object)f).ToArray();
        }

        private List<string> ParseWithoutQuotes(string record)
        {
            var parts = record.Split(_config.FieldSeparator, StringSplitOptions.None);
            return parts.ToList();
        }

        private List<string> ParseWithQuotes(string record)
        {
            var values = new List<string>();
            var builder = new StringBuilder();
            bool inQuotes = false;
            char quoteChar = _config.QuoteStyle switch
            {
                QuoteStyle.DoubleQuote => '"',
                QuoteStyle.SingleQuote => '\'',
                _ => '"'
            };
            var separator = _config.FieldSeparator;

            for (int i = 0; i < record.Length; i++)
            {
                var c = record[i];
                if (c == quoteChar)
                {
                    if (inQuotes && i + 1 < record.Length && record[i + 1] == quoteChar)
                    {
                        builder.Append(c);
                        i++;
                    }
                    else
                    {
                        inQuotes = !inQuotes;
                    }
                }
                else if (!inQuotes && MatchesSeparator(record, i, separator))
                {
                    values.Add(builder.ToString());
                    builder.Clear();
                    i += separator.Length - 1;
                }
                else
                {
                    builder.Append(c);
                }
            }

            values.Add(builder.ToString());
            return values;
        }

        private static bool MatchesSeparator(string record, int index, string separator)
        {
            if (separator.Length == 1)
            {
                return record[index] == separator[0];
            }
            if (index + separator.Length > record.Length)
            {
                return false;
            }
            return string.CompareOrdinal(record, index, separator, 0, separator.Length) == 0;
        }

        private StreamReader OpenReader()
        {
            if (_config.InputPath == "-")
            {
                return new StreamReader(Console.OpenStandardInput(), Encoding.UTF8, detectEncodingFromByteOrderMarks: true, leaveOpen: false);
            }
            var stream = File.OpenRead(_config.InputPath);
            return new StreamReader(stream, Encoding.UTF8);
        }
    }

    public sealed class XmlStreamReader
    {
        private readonly XmlSourceConfig _config;

        public XmlStreamReader(XmlSourceConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            if (string.IsNullOrWhiteSpace(_config.InputPath) && _config.InputStream is null)
                throw new ArgumentException("Either InputPath or InputStream is required", nameof(config));
            if (_config.ExpectedWidth <= 0)
                throw new ArgumentException("ExpectedWidth must be positive", nameof(config));
        }

        public IEnumerable<object[]> Read()
        {
            return _config.RecordSeparator == RecordSeparatorKind.Null
                ? ReadNullSeparated()
                : ReadLineSeparated();
        }

        private IEnumerable<object[]> ReadLineSeparated()
        {
            using var reader = OpenReader();
            var lines = ReadAllLines(reader);
            foreach (var fragment in SplitFragments(lines, '\n'))
            {
                var row = ParseFragment(fragment);
                if (row is not null)
                {
                    yield return row;
                }
            }
        }

        private static IEnumerable<string> ReadAllLines(StreamReader reader)
        {
            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                yield return line;
            }
        }

        private IEnumerable<object[]> ReadNullSeparated()
        {
            using var reader = OpenReader();
            var text = reader.ReadToEnd();
            var parts = text.Split('\0');
            foreach (var fragment in parts)
            {
                var row = ParseFragment(fragment);
                if (row is not null)
                {
                    yield return row;
                }
            }
        }

        private static IEnumerable<string> SplitFragments(IEnumerable<string> lines, char delimiter)
        {
            var buffer = new StringBuilder();
            foreach (var line in lines)
            {
                // Empty line signals end of fragment
                if (line.Length == 0)
                {
                    if (buffer.Length > 0)
                    {
                        yield return buffer.ToString();
                        buffer.Clear();
                    }
                }
                else
                {
                    buffer.AppendLine(line);
                }
            }
            if (buffer.Length > 0)
            {
                yield return buffer.ToString();
            }
        }

        private object[]? ParseFragment(string fragment)
        {
            if (string.IsNullOrWhiteSpace(fragment))
            {
                return null;
            }

            try
            {
                // Inject namespace declarations if needed
                fragment = InjectNamespaces(fragment);

                var doc = XDocument.Parse(fragment, LoadOptions.PreserveWhitespace | LoadOptions.SetLineInfo);
                var map = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
                if (doc.Root is XElement root)
                {
                    // Add the root element's qualified name as the Type
                    var rootPrefix = ResolvePrefix(root.Name);
                    var rootType = !string.IsNullOrEmpty(rootPrefix)
                        ? $"{rootPrefix}:{root.Name.LocalName}"
                        : root.Name.LocalName;
                    map["Type"] = rootType;

                    // Process root element and its children
                    AddElement(map, root);
                }
                if (_config.ExpectedWidth <= 1)
                {
                    return new object[] { map };
                }
                var values = map.Values.Take(_config.ExpectedWidth).ToList();
                while (values.Count < _config.ExpectedWidth)
                {
                    values.Add(null);
                }
                return values.ToArray();
            }
            catch (Exception ex)
            {
                throw new InvalidDataException($"Failed to parse XML fragment: {ex.Message}", ex);
            }
        }

        private string InjectNamespaces(string fragment)
        {
            if (_config.NamespacePrefixes == null || _config.NamespacePrefixes.Count == 0)
            {
                return fragment;
            }

            // Find the first opening tag and inject xmlns declarations
            var tagStart = fragment.IndexOf('<');
            if (tagStart < 0) return fragment;

            var tagEnd = fragment.IndexOf('>', tagStart);
            if (tagEnd < 0) return fragment;

            // Check if this is a self-closing tag
            var isSelfClosing = fragment[tagEnd - 1] == '/';
            var insertPos = isSelfClosing ? tagEnd - 1 : tagEnd;

            // Get the opening tag content to check for existing xmlns declarations
            var tagContent = fragment.Substring(tagStart, tagEnd - tagStart + 1);

            // Build xmlns declarations, skipping ones that already exist
            var xmlnsDecls = new StringBuilder();
            foreach (var kvp in _config.NamespacePrefixes)
            {
                var xmlnsAttr = $"xmlns:{kvp.Value}=";
                if (!tagContent.Contains(xmlnsAttr))
                {
                    xmlnsDecls.Append($" xmlns:{kvp.Value}=\"{kvp.Key}\"");
                }
            }

            // Only insert if we have declarations to add
            if (xmlnsDecls.Length == 0)
            {
                return fragment;
            }

            // Insert the declarations
            return fragment.Insert(insertPos, xmlnsDecls.ToString());
        }

        private void AddElement(IDictionary<string, object?> map, XElement element)
        {
            var local = element.Name.LocalName;
            var qualified = element.Name.ToString();
            var prefix = ResolvePrefix(element.Name);

            var content = ExtractElementText(element);

            if (_config.NestedProjection)
            {
                var nested = BuildNested(element);
                map[local] = nested;
                map[qualified] = nested;
                if (!string.IsNullOrEmpty(prefix))
                {
                    map[$"{prefix}:{local}"] = nested;
                }
            }
            else
            {
                map[local] = content;
                map[qualified] = content;
                if (!string.IsNullOrEmpty(prefix))
                {
                    map[$"{prefix}:{local}"] = content;
                }
            }

            foreach (var attr in element.Attributes())
            {
                var attrLocal = attr.Name.LocalName;
                var attrQualified = attr.Name.ToString();
                var attrPrefix = ResolvePrefix(attr.Name);

                // Element-scoped attribute keys (prevents conflicts when multiple child elements have same attribute)
                // e.g., "seeAlso@rdf:resource" and "parentTree@rdf:resource" are now distinct
                map[$"{local}@{attrLocal}"] = attr.Value;
                map[$"{qualified}@{attrQualified}"] = attr.Value;
                if (!string.IsNullOrEmpty(prefix))
                {
                    map[$"{prefix}:{local}@{attrLocal}"] = attr.Value;
                }
                if (!string.IsNullOrEmpty(attrPrefix))
                {
                    map[$"{local}@{attrPrefix}:{attrLocal}"] = attr.Value;
                    if (!string.IsNullOrEmpty(prefix))
                    {
                        map[$"{prefix}:{local}@{attrPrefix}:{attrLocal}"] = attr.Value;
                    }
                }

                // Global attribute keys (backward compatibility - may conflict if multiple elements have same attribute)
                map[$"@{attrLocal}"] = attr.Value;
                map[$"@{attrQualified}"] = attr.Value;
                if (!string.IsNullOrEmpty(attrPrefix))
                {
                    map[$"@{attrPrefix}:{attrLocal}"] = attr.Value;
                }
            }

            foreach (var child in element.Elements())
            {
                AddElement(map, child);
            }
        }

        private object BuildNested(XElement element)
        {
            if (!element.HasElements)
            {
                return ExtractElementText(element);
            }
            var dict = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
            foreach (var child in element.Elements())
            {
                var key = child.Name.LocalName;
                var value = BuildNested(child);
                if (dict.ContainsKey(key))
                {
                    // If multiple children with same name, store as list
                    if (dict[key] is List<object?> list)
                    {
                        list.Add(value);
                    }
                    else
                    {
                        dict[key] = new List<object?> { dict[key], value };
                    }
                }
                else
                {
                    dict[key] = value;
                }
            }
            foreach (var attr in element.Attributes())
            {
                dict[$"@{attr.Name.LocalName}"] = attr.Value;
            }
            return dict;
        }

        private string ExtractElementText(XElement element)
        {
            // If it's Pearltrees and we see CDATA, keep the CDATA text verbatim
            if (_config.TreatPearltreesCDataAsText)
            {
                var cdata = element.Nodes().OfType<XCData>().FirstOrDefault();
                if (cdata is not null)
                {
                    return cdata.Value;
                }
            }
            return element.Value;
        }

        private string? ResolvePrefix(XName name)
        {
            var ns = name.NamespaceName;
            if (string.IsNullOrEmpty(ns))
            {
                return null;
            }

            if (_config.NamespacePrefixes is { } map && map.TryGetValue(ns, out var mapped))
            {
                return mapped;
            }

            if (XmlDefaults.BuiltinNamespacePrefixes.TryGetValue(ns, out var builtin))
            {
                return builtin;
            }

            return null;
        }

        private StreamReader OpenReader()
        {
            if (_config.InputStream is not null)
            {
                return new StreamReader(_config.InputStream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, leaveOpen: false);
            }
            if (_config.InputPath == "-")
            {
                return new StreamReader(Console.OpenStandardInput(), Encoding.UTF8, detectEncodingFromByteOrderMarks: true, leaveOpen: false);
            }
            var stream = File.OpenRead(_config.InputPath);
            return new StreamReader(stream, Encoding.UTF8);
        }
    }

public enum JsonColumnType
{
    String,
    Integer,
    Long,
    Double,
    Boolean,
    Json
}

public enum JsonSchemaFieldKind
{
    Value,
    Record
}

public enum JsonColumnSelectorKind
{
    Path,
    JsonPath
}

public enum JsonNullPolicy
{
    Allow,
    Fail,
    Skip,
    Default
}

public sealed record JsonColumnSelectorConfig(string Selector, JsonColumnSelectorKind Kind);

public sealed record JsonSchemaFieldConfig(
    string PropertyName,
    string Path,
    JsonColumnSelectorKind SelectorKind,
    JsonColumnType ColumnType,
    JsonSchemaFieldKind FieldKind = JsonSchemaFieldKind.Value,
    string? RecordTypeName = null,
    JsonSchemaFieldConfig[]? NestedFields = null);

public sealed record JsonSourceConfig
{
    public string InputPath { get; init; } = string.Empty;
    public JsonColumnSelectorConfig[] ColumnSelectors { get; init; } = Array.Empty<JsonColumnSelectorConfig>();
    public RecordSeparatorKind RecordSeparator { get; init; } = RecordSeparatorKind.LineFeed;
    public int SkipRows { get; init; } = 0;
    public int ExpectedWidth { get; init; } = 0;
    public bool TreatArrayAsStream { get; init; } = true;
    public string? TargetTypeName { get; init; }
    public bool ReturnObject { get; init; } = false;
    public JsonSchemaFieldConfig[] SchemaFields { get; init; } = Array.Empty<JsonSchemaFieldConfig>();
    public JsonNullPolicy NullPolicy { get; init; } = JsonNullPolicy.Allow;
    public string? NullReplacement { get; init; }
}

    public sealed class JsonStreamReader
    {
        private readonly JsonSourceConfig _config;
        private readonly IJsonSelector[] _selectors;
        private readonly SchemaFieldRuntime[] _schemaFields;
        private readonly bool _returnObject;
        private readonly Type? _targetType;
        private readonly bool _treatArrayAsStream;
        private readonly JsonNullPolicy _nullPolicy;
        private readonly string? _nullReplacement;
        private readonly JsonSerializerOptions _serializerOptions = new()
        {
            PropertyNameCaseInsensitive = true
        };

        public JsonStreamReader(JsonSourceConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            if (string.IsNullOrWhiteSpace(_config.InputPath))
                throw new ArgumentException("InputPath is required", nameof(config));
            _returnObject = _config.ReturnObject;

            _schemaFields = _config.SchemaFields?
                .Select(SchemaFieldRuntime.Create)
                .ToArray() ?? Array.Empty<SchemaFieldRuntime>();
            _selectors = _config.ColumnSelectors?
                .Select(selector => JsonSelectorFactory.Create(selector.Selector, selector.Kind))
                .ToArray() ?? Array.Empty<IJsonSelector>();

            if (_returnObject)
            {
                if (string.IsNullOrWhiteSpace(_config.TargetTypeName))
                {
                    throw new ArgumentException("TargetTypeName is required when ReturnObject is true.", nameof(config));
                }
                _targetType = ResolveTargetType(_config.TargetTypeName!);
            }
            else
            {
                _targetType = null;
            }

            _treatArrayAsStream = _config.TreatArrayAsStream;
            _nullPolicy = _config.NullPolicy;
            _nullReplacement = _config.NullReplacement;

            if (_schemaFields.Length > 0 && !_returnObject)
            {
                throw new ArgumentException("SchemaFields require ReturnObject=true.", nameof(config));
            }

            if (!_returnObject && _selectors.Length == 0)
            {
                throw new ArgumentException("ColumnSelectors must be provided for JSON sources when ReturnObject is false.", nameof(config));
            }
        }

        public IEnumerable<object[]> Read()
        {
            if (!File.Exists(_config.InputPath))
            {
                throw new FileNotFoundException("JSON input not found", _config.InputPath);
            }

            var text = File.ReadAllText(_config.InputPath, Encoding.UTF8);
            var trimmed = text.TrimStart();
            var width = _returnObject
                ? Math.Max(_config.ExpectedWidth, 1)
                : Math.Max(_config.ExpectedWidth, _selectors.Length);
            if (width <= 0)
            {
                width = _returnObject ? 1 : _selectors.Length;
            }
            if (_treatArrayAsStream && trimmed.StartsWith("[", StringComparison.Ordinal))
            {
                foreach (var row in ReadFromArray(text, width))
                {
                    yield return row;
                }
                yield break;
            }

            if (_config.RecordSeparator == RecordSeparatorKind.Null)
            {
                foreach (var row in ReadNullSeparated(trimmed, width))
                {
                    yield return row;
                }
            }
            else
            {
                foreach (var row in ReadLineSeparated(trimmed, width))
                {
                    yield return row;
                }
            }
        }

        private IEnumerable<object[]> ReadFromArray(string text, int width)
        {
            using var document = JsonDocument.Parse(text);
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Array)
            {
                throw new InvalidDataException("JSON input expected an array or stream of objects.");
            }

            var index = 0;
            foreach (var element in root.EnumerateArray())
            {
                if (index++ < _config.SkipRows)
                {
                    continue;
                }
                var row = BuildRow(element, width);
                if (TryProcessRow(row, out var processed))
                {
                    yield return processed;
                }
            }
        }

        private IEnumerable<object[]> ReadLineSeparated(string text, int width)
        {
            using var reader = new StringReader(text);
            string? line;
            var skipped = 0;
            while ((line = reader.ReadLine()) is not null)
            {
                line = line.Trim();
                if (line.Length == 0)
                {
                    continue;
                }
                if (line == "[" || line == "]")
                {
                    continue;
                }
                if (skipped < _config.SkipRows)
                {
                    skipped++;
                    continue;
                }
                var row = ParseRecord(line, width);
                if (TryProcessRow(row, out var processed))
                {
                    yield return processed;
                }
            }
        }

        private IEnumerable<object[]> ReadNullSeparated(string text, int width)
        {
            var records = text.Split('\0');
            var skipped = 0;
            foreach (var record in records)
            {
                if (record.Length == 0)
                {
                    continue;
                }
                if (skipped < _config.SkipRows)
                {
                    skipped++;
                    continue;
                }
                var row = ParseRecord(record, width);
                if (TryProcessRow(row, out var processed))
                {
                    yield return processed;
                }
            }
        }

        private object[] ParseRecord(string record, int width)
        {
            using var document = JsonDocument.Parse(record);
            var element = document.RootElement;
            if (element.ValueKind == JsonValueKind.Array && _treatArrayAsStream)
            {
                // Treat array as stream of objects (rare, but handle gracefully)
                foreach (var item in element.EnumerateArray())
                {
                    return BuildRow(item, width);
                }
            }
            return BuildRow(element, width);
        }

        private object[] BuildRow(JsonElement element, int width)
        {
            if (_schemaFields.Length > 0)
            {
                return new[] { BuildSchemaObject(element) };
            }
            if (_returnObject)
            {
                return new[] { DeserializeToObject(element) };
            }
            return ProjectRow(element, width);
        }

        private object[] ProjectRow(JsonElement element, int width)
        {
            if (_selectors.Length == 0)
            {
                throw new InvalidDataException("JSON columns must be specified.");
            }

            var result = new object[width];
            for (var i = 0; i < _selectors.Length && i < width; i++)
            {
                if (_selectors[i].TryEvaluateElement(element, out var selected))
                {
                    result[i] = ConvertJsonValue(selected);
                }
                else
                {
                    result[i] = null;
                }
            }
            return result;
        }

        private bool TryProcessRow(object[] row, out object[] processed)
        {
            processed = row;
            if (_nullPolicy == JsonNullPolicy.Allow)
            {
                return true;
            }

            var hasNull = false;
            for (var i = 0; i < row.Length; i++)
            {
                if (row[i] is null)
                {
                    hasNull = true;
                    switch (_nullPolicy)
                    {
                        case JsonNullPolicy.Fail:
                            throw new InvalidDataException($"Null value encountered while reading '{_config.InputPath}'.");
                        case JsonNullPolicy.Skip:
                            return false;
                        case JsonNullPolicy.Default:
                            row[i] = _nullReplacement ?? string.Empty;
                            break;
                    }
                }
            }

            return true;
        }

        private object? DeserializeToObject(JsonElement element)
        {
            if (_targetType is null)
            {
                throw new InvalidOperationException("Target type not resolved for object deserialization.");
            }

            var json = element.GetRawText();
            try
            {
                return System.Text.Json.JsonSerializer.Deserialize(json, _targetType, _serializerOptions)
                    ?? throw new InvalidDataException($"Deserialization returned null for type '{_targetType.FullName}'.");
            }
            catch (JsonException ex)
            {
                throw new InvalidDataException($"Failed to deserialize JSON into '{_targetType.FullName}': {ex.Message}", ex);
            }
        }

        private object BuildSchemaObject(JsonElement element)
        {
            if (_targetType is null)
            {
                throw new InvalidOperationException("Target type not resolved for schema deserialization.");
            }

            var values = new object?[_schemaFields.Length];
            for (var i = 0; i < _schemaFields.Length; i++)
            {
                values[i] = _schemaFields[i].GetValue(element);
            }

            var instance = Activator.CreateInstance(_targetType, values);
            if (instance is null)
            {
                throw new InvalidDataException($"Activator could not instantiate '{_targetType.FullName}'.");
            }
            return instance;
        }

        private static object? ConvertJsonValue(JsonElement value)
        {
            return value.ValueKind switch
            {
                JsonValueKind.String => value.GetString(),
                JsonValueKind.Number => value.TryGetInt64(out var number)
                    ? number
                    : value.TryGetDouble(out var dbl) ? dbl : value.GetRawText(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Null => null,
                JsonValueKind.Undefined => null,
                JsonValueKind.Object => value.GetRawText(),
                JsonValueKind.Array => value.GetRawText(),
                _ => value.GetRawText()
            };
        }

        private sealed class SchemaFieldRuntime
        {
            private readonly string _name;
            private readonly IJsonSelector _selector;
            private readonly JsonColumnType _columnType;
            private readonly JsonSchemaFieldKind _fieldKind;
            private readonly Type? _recordType;
            private readonly SchemaFieldRuntime[] _nestedFields;

            private SchemaFieldRuntime(
                string name,
                IJsonSelector selector,
                JsonColumnType columnType,
                JsonSchemaFieldKind fieldKind,
                Type? recordType,
                SchemaFieldRuntime[] nestedFields)
            {
                _name = name;
                _selector = selector;
                _columnType = columnType;
                _fieldKind = fieldKind;
                _recordType = recordType;
                _nestedFields = nestedFields;
            }

            public static SchemaFieldRuntime Create(JsonSchemaFieldConfig config)
            {
                if (config is null) throw new ArgumentNullException(nameof(config));
                var selector = JsonSelectorFactory.Create(config.Path, config.SelectorKind);
                var nested = config.NestedFields?
                    .Select(Create)
                    .ToArray() ?? Array.Empty<SchemaFieldRuntime>();

                Type? recordType = null;
                if (config.FieldKind == JsonSchemaFieldKind.Record)
                {
                    if (string.IsNullOrWhiteSpace(config.RecordTypeName))
                    {
                        throw new InvalidDataException($"Schema field '{config.PropertyName}' is a record but RecordTypeName is missing.");
                    }
                    recordType = ResolveTargetType(config.RecordTypeName);
                }

                return new SchemaFieldRuntime(
                    config.PropertyName,
                    selector,
                    config.ColumnType,
                    config.FieldKind,
                    recordType,
                    nested);
            }

            public object? GetValue(JsonElement element)
            {
                if (_fieldKind == JsonSchemaFieldKind.Record)
                {
                    return TryBuildRecord(element);
                }

                return _selector.TryEvaluateElement(element, out var result)
                    ? ConvertValue(result)
                    : null;
            }

            private object? TryBuildRecord(JsonElement element)
            {
                if (_recordType is null)
                {
                    throw new InvalidDataException($"Schema field '{_name}' is a record but no target type was resolved.");
                }

                if (!_selector.TryEvaluateElement(element, out var subElement))
                {
                    return null;
                }

                var values = new object?[_nestedFields.Length];
                for (var i = 0; i < _nestedFields.Length; i++)
                {
                    values[i] = _nestedFields[i].GetValue(subElement);
                }

                var instance = Activator.CreateInstance(_recordType, values);
                if (instance is null)
                {
                    throw new InvalidDataException($"Activator could not instantiate '{_recordType.FullName}'.");
                }
                return instance;
            }

            private object? ConvertValue(JsonElement element)
            {
                if (element.ValueKind == JsonValueKind.Null || element.ValueKind == JsonValueKind.Undefined)
                {
                    return null;
                }

                try
                {
                    return _columnType switch
                    {
                        JsonColumnType.String => element.ToString(),
                        JsonColumnType.Boolean => element.ValueKind == JsonValueKind.True || element.ValueKind == JsonValueKind.False
                            ? element.GetBoolean()
                            : Convert.ToBoolean(JsonStreamReader.ConvertJsonValue(element) ?? false, CultureInfo.InvariantCulture),
                        JsonColumnType.Integer => element.ValueKind == JsonValueKind.Number && element.TryGetInt32(out var intValue)
                            ? intValue
                            : Convert.ToInt32(JsonStreamReader.ConvertJsonValue(element)!, CultureInfo.InvariantCulture),
                        JsonColumnType.Long => element.ValueKind == JsonValueKind.Number && element.TryGetInt64(out var longValue)
                            ? longValue
                            : Convert.ToInt64(JsonStreamReader.ConvertJsonValue(element)!, CultureInfo.InvariantCulture),
                        JsonColumnType.Double => element.ValueKind == JsonValueKind.Number && element.TryGetDouble(out var doubleValue)
                            ? doubleValue
                            : Convert.ToDouble(JsonStreamReader.ConvertJsonValue(element)!, CultureInfo.InvariantCulture),
                        JsonColumnType.Json => element.GetRawText(),
                        _ => element.ToString()
                    };
                }
                catch (Exception ex)
                {
                    throw new InvalidDataException($"Unable to convert JSON column '{_name}' to {_columnType}: {ex.Message}", ex);
                }
            }
        }

        private interface IJsonSelector
        {
            bool TryEvaluateElement(JsonElement element, out JsonElement result);
        }

        private static class JsonSelectorFactory
        {
            public static IJsonSelector Create(string selector, JsonColumnSelectorKind kind)
            {
                return kind switch
                {
                    JsonColumnSelectorKind.JsonPath => new JsonPathExpression(selector),
                    _ => new ColumnPath(selector)
                };
            }
        }

        private sealed class ColumnPath : IJsonSelector
        {
            private readonly PathStep[] _steps;
            private readonly string _raw;

            public ColumnPath(string raw)
            {
                _raw = raw ?? string.Empty;
                _steps = Parse(_raw);
            }

            public object? Evaluate(JsonElement element)
            {
                return TryEvaluateElement(element, out var result)
                    ? JsonStreamReader.ConvertJsonValue(result)
                    : null;
            }

            public bool TryEvaluateElement(JsonElement element, out JsonElement result)
            {
                result = element;
                if (_steps.Length == 0)
                {
                    return true;
                }

                foreach (var step in _steps)
                {
                    switch (step.Kind)
                    {
                        case PathStepKind.Property:
                            if (result.ValueKind == JsonValueKind.Object &&
                                result.TryGetProperty(step.Property!, out var next))
                            {
                                result = next;
                            }
                            else
                            {
                                result = default;
                                return false;
                            }
                            break;
                        case PathStepKind.Index:
                            if (result.ValueKind == JsonValueKind.Array &&
                                TryGetArrayElement(result, step.Index, out var elementAtIndex))
                            {
                                result = elementAtIndex;
                            }
                            else
                            {
                                result = default;
                                return false;
                            }
                            break;
                        default:
                            result = default;
                            return false;
                    }
                }

                return true;
            }

            private static bool TryGetArrayElement(JsonElement arrayElement, int index, out JsonElement value)
            {
                if (index < 0)
                {
                    value = default;
                    return false;
                }

                var i = 0;
                foreach (var item in arrayElement.EnumerateArray())
                {
                    if (i++ == index)
                    {
                        value = item;
                        return true;
                    }
                }

                value = default;
                return false;
            }

            private static PathStep[] Parse(string path)
            {
                if (string.IsNullOrWhiteSpace(path))
                {
                    return Array.Empty<PathStep>();
                }

                var steps = new List<PathStep>();
                var span = path.AsSpan();
                var i = 0;
                while (i < span.Length)
                {
                    if (span[i] == '.')
                    {
                        i++;
                        continue;
                    }

                    if (span[i] == '[')
                    {
                        if (i + 1 < span.Length && (span[i + 1] == '"' || span[i + 1] == '\''))
                        {
                            var quote = span[i + 1];
                            var j = i + 2;
                            var builder = new StringBuilder();
                            while (j < span.Length && span[j] != quote)
                            {
                                builder.Append(span[j]);
                                j++;
                            }
                            if (j >= span.Length)
                            {
                                throw new InvalidDataException($"Unterminated quoted property in column path '{path}'.");
                            }
                            if (j + 1 >= span.Length || span[j + 1] != ']')
                            {
                                throw new InvalidDataException($"Missing closing ] in column path '{path}'.");
                            }
                            steps.Add(PathStep.ForProperty(builder.ToString()));
                            i = j + 2;
                            continue;
                        }

                        var end = path.IndexOf(']', i + 1);
                        if (end < 0)
                        {
                            break;
                        }
                        var slice = path.Substring(i + 1, end - i - 1).Trim();
                        if (int.TryParse(slice, out var idx))
                        {
                            steps.Add(PathStep.ForIndex(idx));
                        }
                        i = end + 1;
                        continue;
                    }

                    var start = i;
                    while (i < span.Length && span[i] != '.' && span[i] != '[')
                    {
                        i++;
                    }
                    var name = path.Substring(start, i - start).Trim();
                    if (!string.IsNullOrEmpty(name))
                    {
                        steps.Add(PathStep.ForProperty(name));
                    }
                }

                return steps.ToArray();
            }
        }

        private enum PathStepKind
        {
            Property,
            Index
        }

        private readonly struct PathStep
        {
            private PathStep(PathStepKind kind, string? property, int index)
            {
                Kind = kind;
                Property = property;
                Index = index;
            }

            public PathStepKind Kind { get; }
            public string? Property { get; }
            public int Index { get; }

            public static PathStep ForProperty(string name) =>
                new PathStep(PathStepKind.Property, name, -1);

            public static PathStep ForIndex(int index) =>
                new PathStep(PathStepKind.Index, null, index);
        }

        private sealed class JsonPathExpression : IJsonSelector
        {
            private readonly JsonPathToken[] _tokens;

            public JsonPathExpression(string expression)
            {
                if (string.IsNullOrWhiteSpace(expression))
                {
                    throw new ArgumentException("JSONPath selector cannot be empty.", nameof(expression));
                }
                _tokens = JsonPathParser.Parse(expression);
            }

            public bool TryEvaluateElement(JsonElement element, out JsonElement result)
            {
                var matches = JsonPathEvaluator.Evaluate(element, _tokens);
                if (matches.Count > 0)
                {
                    result = matches[0];
                    return true;
                }
                result = default;
                return false;
            }
        }

        private readonly record struct JsonPathToken(JsonPathTokenKind Kind, string? Text = null, int Index = -1);

        private enum JsonPathTokenKind
        {
            Root,
            Property,
            WildcardProperty,
            RecursiveProperty,
            AnyRecursive,
            Index,
            WildcardIndex
        }

        private static class JsonPathParser
        {
            public static JsonPathToken[] Parse(string expression)
            {
                var tokens = new List<JsonPathToken>();
                var span = expression.AsSpan();
                var i = 0;
                while (i < span.Length)
                {
                    var ch = span[i];
                    if (char.IsWhiteSpace(ch))
                    {
                        i++;
                        continue;
                    }

                    if (ch == '$')
                    {
                        tokens.Add(new JsonPathToken(JsonPathTokenKind.Root));
                        i++;
                        continue;
                    }

                    if (ch == '.')
                    {
                        if (i + 1 < span.Length && span[i + 1] == '.')
                        {
                            i += 2;
                            if (i < span.Length && span[i] == '*')
                            {
                                tokens.Add(new JsonPathToken(JsonPathTokenKind.AnyRecursive));
                                i++;
                                continue;
                            }
                            var recursiveName = ReadName(span, ref i);
                            tokens.Add(new JsonPathToken(JsonPathTokenKind.RecursiveProperty, recursiveName));
                        }
                        else
                        {
                            i++;
                            if (i < span.Length && span[i] == '*')
                            {
                                tokens.Add(new JsonPathToken(JsonPathTokenKind.WildcardProperty));
                                i++;
                                continue;
                            }
                            var name = ReadName(span, ref i);
                            tokens.Add(new JsonPathToken(JsonPathTokenKind.Property, name));
                        }
                        continue;
                    }

                    if (ch == '[')
                    {
                        i++;
                        if (i < span.Length && (span[i] == '"' || span[i] == '\''))
                        {
                            var quote = span[i++];
                            var builder = new StringBuilder();
                            while (i < span.Length && span[i] != quote)
                            {
                                builder.Append(span[i]);
                                i++;
                            }
                            if (i >= span.Length)
                            {
                                throw new InvalidDataException($"Unterminated quoted token in JSONPath '{expression}'.");
                            }
                            i++; // skip quote
                            if (i >= span.Length || span[i] != ']')
                            {
                                throw new InvalidDataException($"Missing closing ] in JSONPath '{expression}'.");
                            }
                            i++;
                            tokens.Add(new JsonPathToken(JsonPathTokenKind.Property, builder.ToString()));
                            continue;
                        }
                        var end = expression.IndexOf(']', i);
                        if (end < 0)
                        {
                            throw new InvalidDataException($"Missing closing ] in JSONPath '{expression}'.");
                        }
                        var slice = expression.Substring(i, end - i).Trim();
                        if (slice == "*")
                        {
                            tokens.Add(new JsonPathToken(JsonPathTokenKind.WildcardIndex));
                        }
                        else if (int.TryParse(slice, NumberStyles.Integer, CultureInfo.InvariantCulture, out var idx))
                        {
                            tokens.Add(new JsonPathToken(JsonPathTokenKind.Index, null, idx));
                        }
                        else
                        {
                            tokens.Add(new JsonPathToken(JsonPathTokenKind.Property, slice));
                        }
                        i = end + 1;
                        continue;
                    }

                    var fallback = ReadName(span, ref i);
                    if (!string.IsNullOrEmpty(fallback))
                    {
                        tokens.Add(new JsonPathToken(JsonPathTokenKind.Property, fallback));
                    }
                }

                return tokens.ToArray();
            }

            private static string ReadName(ReadOnlySpan<char> span, ref int index)
            {
                var start = index;
                while (index < span.Length && IsNameChar(span[index]))
                {
                    index++;
                }
                return span.Slice(start, index - start).ToString();
            }

            private static bool IsNameChar(char ch) =>
                char.IsLetterOrDigit(ch) || ch == '_' || ch == '-';
        }

        private static class JsonPathEvaluator
        {
            public static List<JsonElement> Evaluate(JsonElement root, JsonPathToken[] tokens)
            {
                var current = new List<JsonElement> { root };
                foreach (var token in tokens)
                {
                    var next = new List<JsonElement>();
                    foreach (var element in current)
                    {
                        ApplyToken(element, token, next);
                    }
                    current = next;
                    if (current.Count == 0)
                    {
                        break;
                    }
                }
                return current;
            }

            private static void ApplyToken(JsonElement element, JsonPathToken token, List<JsonElement> output)
            {
                switch (token.Kind)
                {
                    case JsonPathTokenKind.Root:
                        output.Add(element);
                        break;
                    case JsonPathTokenKind.Property:
                        if (element.ValueKind == JsonValueKind.Object &&
                            token.Text is not null &&
                            element.TryGetProperty(token.Text, out var property))
                        {
                            output.Add(property);
                        }
                        break;
                    case JsonPathTokenKind.WildcardProperty:
                        if (element.ValueKind == JsonValueKind.Object)
                        {
                            foreach (var prop in element.EnumerateObject())
                            {
                                output.Add(prop.Value);
                            }
                        }
                        break;
                    case JsonPathTokenKind.RecursiveProperty:
                        if (token.Text is not null)
                        {
                            CollectRecursiveProperties(element, token.Text, output);
                        }
                        break;
                    case JsonPathTokenKind.AnyRecursive:
                        CollectAllDescendants(element, output);
                        break;
                case JsonPathTokenKind.Index:
                    if (element.ValueKind == JsonValueKind.Array &&
                        token.Index >= 0 &&
                        TryGetArrayElement(element, token.Index, out var selected))
                    {
                        output.Add(selected);
                    }
                    break;
                    case JsonPathTokenKind.WildcardIndex:
                        if (element.ValueKind == JsonValueKind.Array)
                        {
                            foreach (var child in element.EnumerateArray())
                            {
                                output.Add(child);
                            }
                        }
                        break;
                }
            }

            private static void CollectRecursiveProperties(JsonElement element, string name, List<JsonElement> output)
            {
                if (element.ValueKind == JsonValueKind.Object)
                {
                    if (element.TryGetProperty(name, out var property))
                    {
                        output.Add(property);
                    }
                    foreach (var child in element.EnumerateObject())
                    {
                        CollectRecursiveProperties(child.Value, name, output);
                    }
                }
                else if (element.ValueKind == JsonValueKind.Array)
                {
                    foreach (var child in element.EnumerateArray())
                    {
                        CollectRecursiveProperties(child, name, output);
                    }
                }
            }

        private static void CollectAllDescendants(JsonElement element, List<JsonElement> output)
        {
            output.Add(element);
            if (element.ValueKind == JsonValueKind.Object)
            {
                foreach (var property in element.EnumerateObject())
                {
                    CollectAllDescendants(property.Value, output);
                }
            }
            else if (element.ValueKind == JsonValueKind.Array)
            {
                foreach (var child in element.EnumerateArray())
                {
                    CollectAllDescendants(child, output);
                }
            }
        }

        private static bool TryGetArrayElement(JsonElement arrayElement, int index, out JsonElement value)
        {
            if (index < 0)
            {
                value = default;
                return false;
            }

            var currentIndex = 0;
            foreach (var child in arrayElement.EnumerateArray())
            {
                if (currentIndex == index)
                {
                    value = child;
                    return true;
                }
                currentIndex++;
            }

            value = default;
            return false;
        }
    }

        private static Type ResolveTargetType(string typeName)
        {
            var target = Type.GetType(typeName, throwOnError: false, ignoreCase: false);
            if (target is not null)
            {
                return target;
            }

            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                target = assembly.GetType(typeName, throwOnError: false, ignoreCase: false);
                if (target is not null)
                {
                    return target;
                }
            }

            throw new InvalidOperationException($"Unable to resolve target type '{typeName}'.");
        }
    }
}
