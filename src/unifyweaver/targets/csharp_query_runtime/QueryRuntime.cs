// SPDX-License-Identifier: MIT OR Apache-2.0
// Generated runtime scaffolding for UnifyWeaver C# query execution
// Provides minimal infrastructure for executing declarative plans
// emitted by the forthcoming csharp_query target.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.IO;
using System.Numerics;
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

    public enum RelationRetentionMode
    {
        Streaming,
        Replayable,
        ExternalMaterialized
    }

    public enum PathAwareEdgeRetentionStrategy
    {
        Auto,
        StreamingDirect,
        ReplayableBuffer,
        ExternalMaterialized
    }

    public enum PathAwareSupportRelationRetentionStrategy
    {
        Auto,
        StreamingDirect,
        ReplayableBuffer,
        ExternalMaterialized
    }

    public enum DagRelationRetentionStrategy
    {
        Auto,
        StreamingDirect,
        ReplayableBuffer,
        ExternalMaterialized
    }

    public enum ScanRelationRetentionStrategy
    {
        Auto,
        StreamingDirect,
        ReplayableBuffer,
        ExternalMaterialized
    }

    public enum ClosureRelationRetentionStrategy
    {
        Auto,
        StreamingDirect,
        ReplayableBuffer,
        ExternalMaterialized
    }

    public enum ClosurePairStrategy
    {
        Auto,
        Forward,
        Backward,
        MemoizedBySource,
        MemoizedByTarget,
        MixedDirection,
        MixedDirectionWithPairProbeCache
    }

    public readonly record struct DelimitedRelationSource(
        string InputPath,
        char Delimiter = '	',
        int SkipRows = 1,
        int ExpectedWidth = 2);

    public interface IReplayableRelationSource
    {
        IEnumerable<object[]> Stream();
        List<object[]> Materialize();
    }

    public readonly record struct RelationBinding(
        RelationRetentionMode Mode,
        DelimitedRelationSource? DelimitedSource = null,
        IReplayableRelationSource? ReplayableSource = null);

    public interface IRetentionAwareRelationProvider : IRelationProvider
    {
        bool TryBindRelation(PredicateId predicate, RelationRetentionMode preferredMode, out RelationBinding binding);
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
    /// Propagates group labels from an external seed relation across a plain
    /// transitive closure edge relation, producing (group, reachable) rows.
    /// </summary>
    public sealed record SeedGroupedTransitiveClosureNode(
        PredicateId EdgeRelation,
        PredicateId SeedRelation,
        PredicateId Predicate
    ) : PlanNode;

    /// <summary>
    /// Propagates group labels from an external seed relation across a plain
    /// transitive closure edge relation, producing (group, reachable_count)
    /// rows directly.
    /// </summary>
    public sealed record SeedGroupedTransitiveClosureCountNode(
        PredicateId EdgeRelation,
        PredicateId SeedRelation,
        PredicateId Predicate
    ) : PlanNode;

    /// <summary>
    /// Computes the maximum DAG depth reachable from grouped seed nodes,
    /// producing (group, depth) rows directly.
    /// </summary>
    public sealed record SeedGroupedDagLongestDepthNode(
        PredicateId EdgeRelation,
        PredicateId SeedRelation,
        PredicateId Predicate
    ) : PlanNode;

    /// <summary>
    /// Computes per-group root weight sums over all simple seed-to-root paths,
    /// producing (group, root, weight_sum) rows directly.
    /// </summary>
    public sealed record SeedGroupedPathAwareWeightSumNode(
        PredicateId EdgeRelation,
        PredicateId SeedRelation,
        PredicateId RootRelation,
        PredicateId Predicate,
        double DistanceExponent,
        int MaxDepth = 0
    ) : PlanNode;

    /// <summary>
    /// Computes per-group minimum root depths, producing
    /// (group, root, min_depth) rows directly.
    /// </summary>
    public sealed record SeedGroupedPathAwareDepthMinNode(
        PredicateId EdgeRelation,
        PredicateId SeedRelation,
        PredicateId RootRelation,
        PredicateId Predicate,
        int DirectSeedDepth = 1,
        int MaxDepth = 0
    ) : PlanNode;

    /// <summary>
    /// Computes per-group minimum accumulated root values, producing
    /// (group, root, min_value) rows directly.
    /// </summary>
    public sealed record SeedGroupedPathAwareAccumulationMinNode(
        PredicateId EdgeRelation,
        PredicateId SeedRelation,
        PredicateId RootRelation,
        PredicateId AuxiliaryRelation,
        PredicateId Predicate,
        ArithmeticExpression BaseExpression,
        ArithmeticExpression RecursiveExpression,
        object DirectSeedValue,
        int MaxDepth = 0,
        bool PositiveStepProven = false
    ) : PlanNode;

    /// <summary>
    /// Computes a counted transitive closure while preventing cycles on each
    /// derivation path instead of deduplicating globally by node or tuple.
    /// </summary>
    public sealed record PathAwareTransitiveClosureNode(
        PredicateId EdgeRelation,
        PredicateId Predicate,
        int BaseDepth,
        int DepthIncrement,
        int MaxDepth = 0,
        TableMode AccumulatorMode = TableMode.All
    ) : PlanNode;

    /// <summary>
    /// Computes a path-aware recursive accumulation where each step binds an
    /// auxiliary value from the current source node and updates an accumulator
    /// via arithmetic expressions.
    /// </summary>
    public sealed record PathAwareAccumulationNode(
        PredicateId EdgeRelation,
        PredicateId Predicate,
        PredicateId AuxiliaryRelation,
        ArithmeticExpression BaseExpression,
        ArithmeticExpression RecursiveExpression,
        int MaxDepth = 0,
        TableMode AccumulatorMode = TableMode.All,
        bool PositiveStepProven = false
    ) : PlanNode;

    public enum TableMode
    {
        All,
        Min,
        Max,
        First,
        Sum,
        Count
    }

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

    public enum PathAwareGroupedMinStrategy
    {
        Auto,
        CompactGrouped,
        LegacySeededRows
    }

    public enum PathAwareWeightSumStrategy
    {
        Auto,
        CompactGrouped,
        LegacySeededRows
    }

    internal enum PathAwareGroupedSummaryStrategy
    {
        Auto,
        CompactGrouped,
        LegacySeededRows
    }

    internal enum RelationRetentionPolicyStrategy
    {
        Auto,
        StreamingDirect,
        ReplayableBuffer,
        ExternalMaterialized
    }

    internal enum ScanRelationAccessKind
    {
        Stream,
        List,
        Set
    }

    internal enum ClosureRelationAccessKind
    {
        Edge,
        Support
    }

    internal enum ClosurePairPlanStrategy
    {
        Forward,
        Backward,
        MemoizedBySource,
        MemoizedByTarget,
        MixedDirection,
        MixedDirectionWithPairProbeCache,
        SingleProbeForward,
        SingleProbeBackward
    }

    internal enum PathAwareSupportRelationAccessKind
    {
        Roots,
        Seeds
    }

    internal readonly record struct RelationRetentionSelection(
        RelationRetentionPolicyStrategy Strategy,
        string DecisionMode);

    internal readonly record struct PathAwareEdgeRetentionSelection(
        PathAwareEdgeRetentionStrategy Strategy,
        string DecisionMode);

    internal readonly record struct PathAwareSupportRelationRetentionSelection(
        PathAwareSupportRelationRetentionStrategy Strategy,
        string DecisionMode);

    internal readonly record struct PathAwareGroupedSummarySelection(
        PathAwareGroupedSummaryStrategy Strategy,
        string DecisionMode);

    internal readonly record struct MaterializationPlanSelection(
        RelationRetentionSelection RelationRetention,
        PathAwareGroupedSummarySelection? GroupedSummary,
        string DecisionMode);

    internal readonly record struct DagRelationRetentionSelection(
        DagRelationRetentionStrategy Strategy,
        string DecisionMode);

    internal readonly record struct ScanRelationRetentionSelection(
        ScanRelationRetentionStrategy Strategy,
        string DecisionMode);

    internal readonly record struct ClosureRelationRetentionSelection(
        ClosureRelationRetentionStrategy Strategy,
        string DecisionMode);

    internal readonly record struct ClosurePairStrategySelection(
        ClosurePairPlanStrategy Strategy,
        string DecisionMode);

    public sealed record QueryExecutorOptions(
        bool ReuseCaches = false,
        DagRelationRetentionStrategy DagRelationRetentionStrategy = DagRelationRetentionStrategy.Auto,
        ScanRelationRetentionStrategy ScanRelationRetentionStrategy = ScanRelationRetentionStrategy.Auto,
        ClosureRelationRetentionStrategy ClosureRelationRetentionStrategy = ClosureRelationRetentionStrategy.Auto,
        ClosurePairStrategy ClosurePairStrategy = ClosurePairStrategy.Auto,
        PathAwareEdgeRetentionStrategy PathAwareEdgeRetentionStrategy = PathAwareEdgeRetentionStrategy.Auto,
        PathAwareSupportRelationRetentionStrategy PathAwareSupportRelationRetentionStrategy = PathAwareSupportRelationRetentionStrategy.Auto,
        PathAwareGroupedMinStrategy PathAwareGroupedMinStrategy = PathAwareGroupedMinStrategy.Auto,
        PathAwareWeightSumStrategy PathAwareWeightSumStrategy = PathAwareWeightSumStrategy.Auto,
        int PairProbeCacheMaxEntries = 4096,
        int SeededCacheMaxEntries = 4096,
        int PairProbeCacheAdmissionMinCost = 0,
        double PairProbeCacheAdmissionMinCostPerProbe = 0,
        int SeededCacheAdmissionMinRows = 0,
        double SeededCacheAdmissionMinRowsPerSeed = 0,
        /// <summary>
        /// Maximum number of fixpoint iterations before termination.
        /// Prevents non-convergence on cyclic graphs with counter-bearing
        /// recursive predicates (e.g., hop counting in transitive closure).
        /// Default 0 = unlimited (standard Datalog convergence assumed).
        /// Recommended: 50 for graphs with cycles and arithmetic counters.
        /// </summary>
        int MaxFixpointIterations = 0,
        bool EnableMeasuredClosurePairStrategy = true,
        bool UseSeededClosureCachesForPairBatches = false);

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
        long Builds,
        long Evictions,
        long Admissions,
        long AdmissionSkips
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

    public sealed record QueryPhaseTrace(
        int NodeId,
        string NodeType,
        string Phase,
        TimeSpan Elapsed
    );

    public sealed record QueryMetricTrace(
        int NodeId,
        string NodeType,
        string Metric,
        double Value
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

            public long Evictions;

            public long Admissions;

            public long AdmissionSkips;
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
        private readonly Dictionary<(PlanNode Node, string Phase), TimeSpan> _phases = new(new PlanNodeStrategyComparer());
        private readonly Dictionary<(PlanNode Node, string Metric), double> _metrics = new(new PlanNodeStrategyComparer());
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

        internal void RecordCacheEviction(string cache, string key, long count = 1)
        {
            if (cache is null) throw new ArgumentNullException(nameof(cache));
            if (key is null) throw new ArgumentNullException(nameof(key));
            if (count <= 0)
            {
                return;
            }

            var stats = GetOrAdd(cache, key);
            stats.Evictions += count;
        }

        internal void RecordCacheAdmission(string cache, string key, bool admitted, long count = 1)
        {
            if (cache is null) throw new ArgumentNullException(nameof(cache));
            if (key is null) throw new ArgumentNullException(nameof(key));
            if (count <= 0)
            {
                return;
            }

            var stats = GetOrAdd(cache, key);
            if (admitted)
            {
                stats.Admissions += count;
            }
            else
            {
                stats.AdmissionSkips += count;
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

        internal void RecordPhase(PlanNode node, string phase, TimeSpan elapsed)
        {
            if (node is null) throw new ArgumentNullException(nameof(node));
            if (phase is null) throw new ArgumentNullException(nameof(phase));
            if (elapsed <= TimeSpan.Zero)
            {
                return;
            }

            _ = GetOrAdd(node);
            var key = (node, phase);
            if (_phases.TryGetValue(key, out var existing))
            {
                _phases[key] = existing + elapsed;
            }
            else
            {
                _phases.Add(key, elapsed);
            }
        }

        internal void RecordMetric(PlanNode node, string metric, double value)
        {
            if (node is null) throw new ArgumentNullException(nameof(node));
            if (metric is null) throw new ArgumentNullException(nameof(metric));

            _ = GetOrAdd(node);
            _metrics[(node, metric)] = value;
        }

        internal void AddMetric(PlanNode node, string metric, double value)
        {
            if (node is null) throw new ArgumentNullException(nameof(node));
            if (metric is null) throw new ArgumentNullException(nameof(metric));

            _ = GetOrAdd(node);
            var key = (node, metric);
            if (_metrics.TryGetValue(key, out var existing))
            {
                _metrics[key] = existing + value;
            }
            else
            {
                _metrics.Add(key, value);
            }
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
                        stats.Builds,
                        stats.Evictions,
                        stats.Admissions,
                        stats.AdmissionSkips);
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

        public IReadOnlyList<QueryPhaseTrace> SnapshotPhases()
        {
            return _phases
                .Select(kvp =>
                {
                    var node = kvp.Key.Node;
                    return new QueryPhaseTrace(
                        GetOrAdd(node).Id,
                        node.GetType().Name,
                        kvp.Key.Phase,
                        kvp.Value);
                })
                .OrderBy(s => s.NodeId)
                .ThenBy(s => s.Phase, StringComparer.Ordinal)
                .ToList();
        }

        public IReadOnlyList<QueryMetricTrace> SnapshotMetrics()
        {
            return _metrics
                .Select(kvp =>
                {
                    var node = kvp.Key.Node;
                    return new QueryMetricTrace(
                        GetOrAdd(node).Id,
                        node.GetType().Name,
                        kvp.Key.Metric,
                        kvp.Value);
                })
                .OrderBy(s => s.NodeId)
                .ThenBy(s => s.Metric, StringComparer.Ordinal)
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
                        .Append(" evictions=")
                        .Append(cache.Evictions)
                        .Append(" admissions=")
                        .Append(cache.Admissions)
                        .Append(" admission_skips=")
                        .Append(cache.AdmissionSkips)
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
                    case SeedGroupedTransitiveClosureNode:
                    case SeedGroupedTransitiveClosureCountNode:
                    case SeedGroupedDagLongestDepthNode:
                    case PathAwareTransitiveClosureNode:
                    case PathAwareAccumulationNode:
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
                SeedGroupedTransitiveClosureNode closure => $"SeedGroupedTransitiveClosure edge={closure.EdgeRelation} seeds={closure.SeedRelation}",
                SeedGroupedTransitiveClosureCountNode closure => $"SeedGroupedTransitiveClosureCount edge={closure.EdgeRelation} seeds={closure.SeedRelation}",
                SeedGroupedDagLongestDepthNode closure => $"SeedGroupedDagLongestDepth edge={closure.EdgeRelation} seeds={closure.SeedRelation}",
                SeedGroupedPathAwareWeightSumNode closure => $"SeedGroupedPathAwareWeightSum edge={closure.EdgeRelation} seeds={closure.SeedRelation} roots={closure.RootRelation} exponent={closure.DistanceExponent.ToString(CultureInfo.InvariantCulture)} maxDepth={closure.MaxDepth}",
                SeedGroupedPathAwareDepthMinNode closure => $"SeedGroupedPathAwareDepthMin edge={closure.EdgeRelation} seeds={closure.SeedRelation} roots={closure.RootRelation} directSeedDepth={closure.DirectSeedDepth} maxDepth={closure.MaxDepth}",
                SeedGroupedPathAwareAccumulationMinNode closure => $"SeedGroupedPathAwareAccumulationMin edge={closure.EdgeRelation} seeds={closure.SeedRelation} roots={closure.RootRelation} aux={closure.AuxiliaryRelation} maxDepth={closure.MaxDepth} positiveStep={closure.PositiveStepProven}",
                PathAwareTransitiveClosureNode closure => $"PathAwareTransitiveClosure edge={closure.EdgeRelation} base={closure.BaseDepth} increment={closure.DepthIncrement} maxDepth={closure.MaxDepth} mode={closure.AccumulatorMode}",
                PathAwareAccumulationNode closure => $"PathAwareAccumulation edge={closure.EdgeRelation} aux={closure.AuxiliaryRelation} maxDepth={closure.MaxDepth} mode={closure.AccumulatorMode} positiveStep={closure.PositiveStepProven}",
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
    internal static class DelimitedRelationReader
    {
        public static IEnumerable<object[]> ReadRows(DelimitedRelationSource source)
        {
            using var reader = OpenSequentialReader(source.InputPath);
            for (var i = 0; i < source.SkipRows; i++)
            {
                if (reader.ReadLine() is null)
                {
                    yield break;
                }
            }

            var expectedWidth = Math.Max(2, source.ExpectedWidth);
            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                if (!TrySplitDelimitedLine(line, source.Delimiter, expectedWidth, out var fields))
                {
                    continue;
                }

                yield return fields;
            }
        }

        public static StreamReader OpenSequentialReader(string path) =>
            new(new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.SequentialScan), Encoding.UTF8, detectEncodingFromByteOrderMarks: true, bufferSize: 1 << 16);

        public static bool TrySplitDelimitedLine(string line, char delimiter, int expectedWidth, out object[] fields)
        {
            if (expectedWidth <= 1)
            {
                expectedWidth = 2;
            }

            var parts = new object[expectedWidth];
            var span = line.AsSpan();
            var start = 0;

            for (var i = 0; i < expectedWidth - 1; i++)
            {
                var remainder = span.Slice(start);
                var split = remainder.IndexOf(delimiter);
                if (split <= 0)
                {
                    fields = Array.Empty<object>();
                    return false;
                }

                parts[i] = remainder.Slice(0, split).ToString();
                start += split + 1;
                if (start >= span.Length)
                {
                    fields = Array.Empty<object>();
                    return false;
                }
            }

            var tail = span.Slice(start);
            if (tail.Length == 0 || tail.IndexOf(delimiter) >= 0)
            {
                fields = Array.Empty<object>();
                return false;
            }

            parts[expectedWidth - 1] = tail.ToString();
            fields = parts;
            return true;
        }

        public static bool TrySplitTwoColumnLine(string line, char delimiter, out string left, out string right)
        {
            if (!TrySplitDelimitedLine(line, delimiter, 2, out var fields))
            {
                left = string.Empty;
                right = string.Empty;
                return false;
            }

            left = fields[0]?.ToString() ?? string.Empty;
            right = fields[1]?.ToString() ?? string.Empty;
            return true;
        }
    }

    internal sealed class ReplayableRelationSource : IReplayableRelationSource
    {
        private readonly Func<IEnumerable<object[]>> _factory;
        private readonly object _gate = new();
        private List<object[]>? _buffer;

        public ReplayableRelationSource(Func<IEnumerable<object[]>> factory)
        {
            _factory = factory ?? throw new ArgumentNullException(nameof(factory));
        }

        public bool IsMaterialized => _buffer is not null;

        public IEnumerable<object[]> Stream() => Materialize();

        public List<object[]> Materialize()
        {
            if (_buffer is not null)
            {
                return _buffer;
            }

            lock (_gate)
            {
                _buffer ??= _factory().ToList();
                return _buffer;
            }
        }

        public List<object[]> ProbeMaterialize(int maxRows)
        {
            if (_buffer is not null)
            {
                return _buffer.Take(Math.Max(0, maxRows)).ToList();
            }

            return _factory().Take(Math.Max(0, maxRows)).ToList();
        }
    }

    internal sealed class PathAwareSuccessorBucket
    {
        public PathAwareSuccessorBucket(object? source)
        {
            Source = source;
        }

        public object? Source { get; }

        public List<object?> Targets { get; } = new();
    }

    internal sealed class PathAwareEdgeState
    {
        public PathAwareEdgeState(
            IReadOnlyDictionary<object, PathAwareSuccessorBucket> successors,
            IReadOnlyList<object?> seeds)
        {
            Successors = successors;
            Seeds = seeds;
        }

        public IReadOnlyDictionary<object, PathAwareSuccessorBucket> Successors { get; }

        public IReadOnlyList<object?> Seeds { get; }
    }

    internal sealed class PathAwareSccGraph
    {
        public PathAwareSccGraph(
            IReadOnlyDictionary<object, int> componentByNode,
            int nodeCount,
            int edgeCount,
            int componentCount,
            int cyclicComponentCount,
            int largestComponentSize,
            int largestCyclicComponentSize,
            int condensedEdgeCount)
        {
            ComponentByNode = componentByNode;
            NodeCount = nodeCount;
            EdgeCount = edgeCount;
            ComponentCount = componentCount;
            CyclicComponentCount = cyclicComponentCount;
            LargestComponentSize = largestComponentSize;
            LargestCyclicComponentSize = largestCyclicComponentSize;
            CondensedEdgeCount = condensedEdgeCount;
        }

        public IReadOnlyDictionary<object, int> ComponentByNode { get; }

        public int NodeCount { get; }

        public int EdgeCount { get; }

        public int ComponentCount { get; }

        public int CyclicComponentCount { get; }

        public int LargestComponentSize { get; }

        public int LargestCyclicComponentSize { get; }

        public int CondensedEdgeCount { get; }
    }

    internal readonly record struct SccCondensedWeightedMinStats(
        long LocalStatesExplored,
        long OuterDagStatesExplored,
        long QueuePops);

    internal readonly record struct SccCondensedWeightedMinProbe(
        TimeSpan Elapsed,
        SccCondensedWeightedMinStats Stats);

    internal enum AdditiveMinStepSafety
    {
        StrictlyPositive,
        NonNegative
    }

    public sealed class InMemoryRelationProvider : IRetentionAwareRelationProvider
    {
        private readonly Dictionary<PredicateId, List<object[]>> _store = new();
        private readonly Dictionary<PredicateId, DelimitedRelationSource> _delimitedSources = new();
        private readonly Dictionary<PredicateId, IReplayableRelationSource> _replayableSources = new();

        public void RegisterDelimitedSource(PredicateId predicate, DelimitedRelationSource source)
        {
            _delimitedSources[predicate] = source;
            _replayableSources.Remove(predicate);
        }

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
            _replayableSources.Remove(predicate);
        }

        public IEnumerable<object[]> GetFacts(PredicateId predicate)
        {
            if (_store.TryGetValue(predicate, out var list))
            {
                return list;
            }
            if (_delimitedSources.TryGetValue(predicate, out var source))
            {
                return DelimitedRelationReader.ReadRows(source);
            }
            return Array.Empty<object[]>();
        }

        public bool TryBindRelation(PredicateId predicate, RelationRetentionMode preferredMode, out RelationBinding binding)
        {
            if (preferredMode == RelationRetentionMode.Streaming &&
                _delimitedSources.TryGetValue(predicate, out var streamingSource))
            {
                binding = new RelationBinding(RelationRetentionMode.Streaming, streamingSource);
                return true;
            }

            if (preferredMode == RelationRetentionMode.Replayable)
            {
                if (!_replayableSources.TryGetValue(predicate, out var replayableSource))
                {
                    if (_delimitedSources.TryGetValue(predicate, out var replayableDelimitedSource))
                    {
                        replayableSource = new ReplayableRelationSource(() => DelimitedRelationReader.ReadRows(replayableDelimitedSource));
                        _replayableSources[predicate] = replayableSource;
                    }
                    else if (_store.TryGetValue(predicate, out var replayableList))
                    {
                        replayableSource = new ReplayableRelationSource(() => replayableList);
                        _replayableSources[predicate] = replayableSource;
                    }
                }

                if (replayableSource is not null)
                {
                    binding = new RelationBinding(RelationRetentionMode.Replayable, ReplayableSource: replayableSource);
                    return true;
                }
            }

            if (preferredMode == RelationRetentionMode.ExternalMaterialized && _store.ContainsKey(predicate))
            {
                binding = new RelationBinding(RelationRetentionMode.ExternalMaterialized);
                return true;
            }

            binding = default;
            return false;
        }
    }

    public sealed class QueryExecutor
    {
        private readonly IRelationProvider _provider;
        private static readonly object NullFactIndexKey = new();
        private static readonly RowWrapperComparer StructuralRowWrapperComparer = new(StructuralArrayComparer.Instance);
        private readonly EvaluationContext? _cacheContext;
        private readonly int _pairProbeCacheMaxEntries;
        private readonly int _seededCacheMaxEntries;
        private readonly int _pairProbeCacheAdmissionMinCost;
        private readonly double _pairProbeCacheAdmissionMinCostPerProbe;
        private readonly int _seededCacheAdmissionMinRows;
        private readonly double _seededCacheAdmissionMinRowsPerSeed;
        private readonly int _maxFixpointIterations;
        private readonly DagRelationRetentionStrategy _dagRelationRetentionStrategy;
        private readonly ScanRelationRetentionStrategy _scanRelationRetentionStrategy;
        private readonly ClosureRelationRetentionStrategy _closureRelationRetentionStrategy;
        private readonly ClosurePairStrategy _closurePairStrategy;
        private readonly bool _enableMeasuredClosurePairStrategy;
        private readonly bool _useSeededClosureCachesForPairBatches;
        private readonly PathAwareEdgeRetentionStrategy _pathAwareEdgeRetentionStrategy;
        private readonly PathAwareSupportRelationRetentionStrategy _pathAwareSupportRelationRetentionStrategy;
        private readonly PathAwareGroupedSummaryStrategy _pathAwareGroupedMinStrategy;
        private readonly PathAwareGroupedSummaryStrategy _pathAwareWeightSumStrategy;

        public QueryExecutor(IRelationProvider provider, QueryExecutorOptions? options = null)
        {
            _provider = provider ?? throw new ArgumentNullException(nameof(provider));
            options ??= new QueryExecutorOptions();
            _cacheContext = options.ReuseCaches ? new EvaluationContext() : null;
            _pairProbeCacheMaxEntries = Math.Max(0, options.PairProbeCacheMaxEntries);
            _seededCacheMaxEntries = Math.Max(0, options.SeededCacheMaxEntries);
            _pairProbeCacheAdmissionMinCost = Math.Max(0, options.PairProbeCacheAdmissionMinCost);
            _pairProbeCacheAdmissionMinCostPerProbe = Math.Max(0d, options.PairProbeCacheAdmissionMinCostPerProbe);
            _seededCacheAdmissionMinRows = Math.Max(0, options.SeededCacheAdmissionMinRows);
            _seededCacheAdmissionMinRowsPerSeed = Math.Max(0d, options.SeededCacheAdmissionMinRowsPerSeed);
            _maxFixpointIterations = Math.Max(0, options.MaxFixpointIterations);
            _dagRelationRetentionStrategy = options.DagRelationRetentionStrategy;
            _scanRelationRetentionStrategy = options.ScanRelationRetentionStrategy;
            _closureRelationRetentionStrategy = options.ClosureRelationRetentionStrategy;
            _closurePairStrategy = options.ClosurePairStrategy;
            _enableMeasuredClosurePairStrategy = options.EnableMeasuredClosurePairStrategy;
            _useSeededClosureCachesForPairBatches = options.UseSeededClosureCachesForPairBatches;
            _pathAwareEdgeRetentionStrategy = options.PathAwareEdgeRetentionStrategy;
            _pathAwareSupportRelationRetentionStrategy = options.PathAwareSupportRelationRetentionStrategy;
            _pathAwareGroupedMinStrategy = ToPathAwareGroupedSummaryStrategy(options.PathAwareGroupedMinStrategy);
            _pathAwareWeightSumStrategy = ToPathAwareGroupedSummaryStrategy(options.PathAwareWeightSumStrategy);
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

                if (plan.Root is SeedGroupedTransitiveClosureNode seedGroupedClosure)
                {
                    var rows = ExecuteSeedGroupedTransitiveClosure(seedGroupedClosure, context);
                    var contextCancellationToken = context.CancellationToken;
                    if (contextCancellationToken.CanBeCanceled)
                    {
                        rows = WithCancellation(rows, contextCancellationToken);
                    }

                    var activeTrace = context.Trace;
                    activeTrace?.RecordInvocation(seedGroupedClosure);
                    return activeTrace is null ? rows : activeTrace.WrapEnumeration(seedGroupedClosure, rows);
                }

                if (plan.Root is SeedGroupedTransitiveClosureCountNode seedGroupedCount)
                {
                    var rows = ExecuteSeedGroupedTransitiveClosureCount(seedGroupedCount, context);
                    var contextCancellationToken = context.CancellationToken;
                    if (contextCancellationToken.CanBeCanceled)
                    {
                        rows = WithCancellation(rows, contextCancellationToken);
                    }

                    var activeTrace = context.Trace;
                    activeTrace?.RecordInvocation(seedGroupedCount);
                    return activeTrace is null ? rows : activeTrace.WrapEnumeration(seedGroupedCount, rows);
                }

                if (plan.Root is SeedGroupedDagLongestDepthNode seedGroupedDepth)
                {
                    var rows = ExecuteSeedGroupedDagLongestDepth(seedGroupedDepth, context);
                    var contextCancellationToken = context.CancellationToken;
                    if (contextCancellationToken.CanBeCanceled)
                    {
                        rows = WithCancellation(rows, contextCancellationToken);
                    }

                    var activeTrace = context.Trace;
                    activeTrace?.RecordInvocation(seedGroupedDepth);
                    return activeTrace is null ? rows : activeTrace.WrapEnumeration(seedGroupedDepth, rows);
                }

                if (plan.Root is SeedGroupedPathAwareWeightSumNode seedGroupedPathAwareWeightSum)
                {
                    var rows = ExecuteSeedGroupedPathAwareWeightSum(seedGroupedPathAwareWeightSum, context);
                    var contextCancellationToken = context.CancellationToken;
                    if (contextCancellationToken.CanBeCanceled)
                    {
                        rows = WithCancellation(rows, contextCancellationToken);
                    }

                    var activeTrace = context.Trace;
                    activeTrace?.RecordInvocation(seedGroupedPathAwareWeightSum);
                    return activeTrace is null ? rows : activeTrace.WrapEnumeration(seedGroupedPathAwareWeightSum, rows);
                }

                if (plan.Root is SeedGroupedPathAwareDepthMinNode seedGroupedPathAwareDepthMin)
                {
                    var rows = ExecuteSeedGroupedPathAwareDepthMin(seedGroupedPathAwareDepthMin, context);
                    var contextCancellationToken = context.CancellationToken;
                    if (contextCancellationToken.CanBeCanceled)
                    {
                        rows = WithCancellation(rows, contextCancellationToken);
                    }

                    var activeTrace = context.Trace;
                    activeTrace?.RecordInvocation(seedGroupedPathAwareDepthMin);
                    return activeTrace is null ? rows : activeTrace.WrapEnumeration(seedGroupedPathAwareDepthMin, rows);
                }

                if (plan.Root is SeedGroupedPathAwareAccumulationMinNode seedGroupedPathAwareAccumulationMin)
                {
                    var rows = ExecuteSeedGroupedPathAwareAccumulationMin(seedGroupedPathAwareAccumulationMin, context);
                    var contextCancellationToken = context.CancellationToken;
                    if (contextCancellationToken.CanBeCanceled)
                    {
                        rows = WithCancellation(rows, contextCancellationToken);
                    }

                    var activeTrace = context.Trace;
                    activeTrace?.RecordInvocation(seedGroupedPathAwareAccumulationMin);
                    return activeTrace is null ? rows : activeTrace.WrapEnumeration(seedGroupedPathAwareAccumulationMin, rows);
                }

                if (plan.Root is PathAwareTransitiveClosureNode pathAwareClosure)
                {
                    IEnumerable<object[]> rows;
                    if (inputPositions.Count > 0 && inputPositions[0] == 0)
                    {
                        rows = ExecuteSeededPathAwareTransitiveClosure(pathAwareClosure, inputPositions, paramList, context);
                        if (!(inputPositions.Count == 1 && inputPositions[0] == 0))
                        {
                            rows = FilterByParameters(rows, inputPositions, paramList);
                        }
                    }
                    else
                    {
                        rows = FilterByParameters(ExecutePathAwareTransitiveClosure(pathAwareClosure, context), inputPositions, paramList);
                    }

                    var contextCancellationToken = context.CancellationToken;
                    if (contextCancellationToken.CanBeCanceled)
                    {
                        rows = WithCancellation(rows, contextCancellationToken);
                    }

                    var activeTrace = context.Trace;
                    activeTrace?.RecordInvocation(pathAwareClosure);
                    return activeTrace is null ? rows : activeTrace.WrapEnumeration(pathAwareClosure, rows);
                }

                if (plan.Root is PathAwareAccumulationNode pathAwareAccumulation)
                {
                    IEnumerable<object[]> rows;
                    if (inputPositions.Count > 0 && inputPositions[0] == 0)
                    {
                        rows = ExecuteSeededPathAwareAccumulation(pathAwareAccumulation, inputPositions, paramList, context);
                        if (!(inputPositions.Count == 1 && inputPositions[0] == 0))
                        {
                            rows = FilterByParameters(rows, inputPositions, paramList);
                        }
                    }
                    else
                    {
                        rows = FilterByParameters(ExecutePathAwareAccumulation(pathAwareAccumulation, context), inputPositions, paramList);
                    }

                    var contextCancellationToken = context.CancellationToken;
                    if (contextCancellationToken.CanBeCanceled)
                    {
                        rows = WithCancellation(rows, contextCancellationToken);
                    }

                    var activeTrace = context.Trace;
                    activeTrace?.RecordInvocation(pathAwareAccumulation);
                    return activeTrace is null ? rows : activeTrace.WrapEnumeration(pathAwareAccumulation, rows);
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
            _cacheContext.ReplayableFactSources.Clear();
            _cacheContext.ScanRelationRetentionSelections.Clear();
            _cacheContext.ClosureRelationRetentionSelections.Clear();
            _cacheContext.PathAwareSupportRelationRetentionSelections.Clear();
            _cacheContext.PathAwareEdgeStates.Clear();
            _cacheContext.PathAwareEdgeRetentionSelections.Clear();
            _cacheContext.FactSets.Clear();
            _cacheContext.FactIndices.Clear();
            _cacheContext.JoinIndices.Clear();
            _cacheContext.TransitiveClosureResults.Clear();
            _cacheContext.TransitiveClosureSeededResults.Clear();
            _cacheContext.TransitiveClosureSeededByTargetResults.Clear();
            _cacheContext.TransitiveClosurePairProbeResults.Clear();
            _cacheContext.GroupedTransitiveClosureResults.Clear();
            _cacheContext.SeedGroupedTransitiveClosureCountResults.Clear();
            _cacheContext.SeedGroupedDagLongestDepthResults.Clear();
            _cacheContext.SeedGroupedPathAwareWeightSumResults.Clear();
            _cacheContext.SeedGroupedPathAwareDepthMinResults.Clear();
            _cacheContext.SeedGroupedPathAwareAccumulationMinResults.Clear();
            _cacheContext.GroupedTransitiveClosureSeededResults.Clear();
            _cacheContext.GroupedTransitiveClosureSeededByTargetResults.Clear();
            _cacheContext.GroupedTransitiveClosurePairProbeResults.Clear();
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
                        ? GetFactStream(scan.Relation, context)
                        : GetScanFactStream(scan.Relation, context, scan);
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

                case SeedGroupedTransitiveClosureNode closure:
                    result = ExecuteSeedGroupedTransitiveClosure(closure, context);
                    break;

                case SeedGroupedTransitiveClosureCountNode closure:
                    result = ExecuteSeedGroupedTransitiveClosureCount(closure, context);
                    break;

                case SeedGroupedDagLongestDepthNode closure:
                    result = ExecuteSeedGroupedDagLongestDepth(closure, context);
                    break;

                case SeedGroupedPathAwareWeightSumNode closure:
                    result = ExecuteSeedGroupedPathAwareWeightSum(closure, context);
                    break;

                case SeedGroupedPathAwareDepthMinNode closure:
                    result = ExecuteSeedGroupedPathAwareDepthMin(closure, context);
                    break;

                case SeedGroupedPathAwareAccumulationMinNode closure:
                    result = ExecuteSeedGroupedPathAwareAccumulationMin(closure, context);
                    break;

                case PathAwareTransitiveClosureNode closure:
                    result = ExecutePathAwareTransitiveClosure(closure, context);
                    break;

                case PathAwareAccumulationNode closure:
                    result = ExecutePathAwareAccumulation(closure, context);
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
                return GetFactStream(scan.Relation, context).Where(tuple =>
                    tuple is not null && TupleMatchesPattern(tuple, scan.Pattern));
            }

            var facts = GetScanFactsList(scan.Relation, context, scan);
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

                case GroupedTransitiveClosureNode closure:
                    predicates.Add(closure.EdgeRelation);
                    return;

                case SeedGroupedTransitiveClosureNode closure:
                    predicates.Add(closure.EdgeRelation);
                    predicates.Add(closure.SeedRelation);
                    return;

                case SeedGroupedTransitiveClosureCountNode closure:
                    predicates.Add(closure.EdgeRelation);
                    predicates.Add(closure.SeedRelation);
                    return;

                case SeedGroupedDagLongestDepthNode closure:
                    predicates.Add(closure.EdgeRelation);
                    predicates.Add(closure.SeedRelation);
                    return;

                case SeedGroupedPathAwareWeightSumNode closure:
                    predicates.Add(closure.EdgeRelation);
                    predicates.Add(closure.SeedRelation);
                    predicates.Add(closure.RootRelation);
                    return;

                case SeedGroupedPathAwareDepthMinNode closure:
                    predicates.Add(closure.EdgeRelation);
                    predicates.Add(closure.SeedRelation);
                    predicates.Add(closure.RootRelation);
                    return;

                case SeedGroupedPathAwareAccumulationMinNode closure:
                    predicates.Add(closure.EdgeRelation);
                    predicates.Add(closure.SeedRelation);
                    predicates.Add(closure.RootRelation);
                    predicates.Add(closure.AuxiliaryRelation);
                    return;

                case PathAwareTransitiveClosureNode closure:
                    predicates.Add(closure.EdgeRelation);
                    return;

                case PathAwareAccumulationNode closure:
                    predicates.Add(closure.EdgeRelation);
                    predicates.Add(closure.AuxiliaryRelation);
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

            foreach (var key in context.TransitiveClosurePairProbeResults.Keys
                         .Where(key => key.EdgeRelation.Equals(predicate) || key.Predicate.Equals(predicate))
                         .ToList())
            {
                context.TransitiveClosurePairProbeResults.Remove(key);
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

            foreach (var key in context.GroupedTransitiveClosurePairProbeResults.Keys
                         .Where(key => key.EdgeRelation.Equals(predicate) || key.Predicate.Equals(predicate))
                         .ToList())
            {
                context.GroupedTransitiveClosurePairProbeResults.Remove(key);
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
                        upperBound = GetScanFactsList(scan.Relation, context, scan).Count;
                        return true;

                    case PatternScanNode scan:
                    {
                        var facts = GetScanFactsList(scan.Relation, context, scan);
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

            static int EstimateFilteredRecursiveRowCount(
                IReadOnlyList<object[]> rows,
                Func<object[], bool>? filter)
            {
                if (filter is null)
                {
                    return rows.Count;
                }

                var count = 0;
                foreach (var row in rows)
                {
                    if (row is not null && filter(row))
                    {
                        count++;
                    }
                }

                return count;
            }

            static int EstimateDistinctRecursiveJoinKeyCount(
                IReadOnlyList<object[]> rows,
                Func<object[], bool>? filter,
                IReadOnlyList<int> keyIndices)
            {
                if (rows.Count == 0 || keyIndices.Count == 0)
                {
                    return 0;
                }

                const int MaxSampleRows = 128;
                var seenKeys = new HashSet<RowWrapper>(StructuralRowWrapperComparer);
                var sampledRows = 0;

                foreach (var row in rows)
                {
                    if (row is null)
                    {
                        continue;
                    }

                    if (filter is not null && !filter(row))
                    {
                        continue;
                    }

                    var key = new object[keyIndices.Count];
                    for (var i = 0; i < keyIndices.Count; i++)
                    {
                        key[i] = GetLookupKey(row, keyIndices[i], NullFactIndexKey);
                    }

                    seenKeys.Add(new RowWrapper(key));
                    sampledRows++;
                    if (sampledRows >= MaxSampleRows)
                    {
                        break;
                    }
                }

                return seenKeys.Count;
            }

            static int ComputeAdaptiveTinyProbeUpperBound(
                int defaultUpperBound,
                int minUpperBound,
                int maxUpperBound,
                int estimatedBuildRows,
                int? estimatedProbeUpperBound)
            {
                var fallback = Math.Clamp(defaultUpperBound, minUpperBound, maxUpperBound);
                if (estimatedBuildRows <= 0 || estimatedProbeUpperBound is null || estimatedProbeUpperBound.Value <= 0)
                {
                    return fallback;
                }

                var ratio = (double)estimatedBuildRows / Math.Max(1, estimatedProbeUpperBound.Value);
                var scale = Math.Sqrt(Math.Max(0.25, ratio));
                var candidate = (int)Math.Round(defaultUpperBound * scale);
                return Math.Clamp(candidate, minUpperBound, maxUpperBound);
            }

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
                        var leftFacts = GetScanFactsList(leftScanPredicate, context, join);
                        var rightFacts = GetScanFactsList(rightScanPredicate, context, join);
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

                        int EstimateDistinctScanJoinKeyCount(
                            PredicateId predicate,
                            IReadOnlyList<object[]> facts,
                            object[]? pattern,
                            Func<object[], bool>? filter,
                            IReadOnlyList<int> keyIndices,
                            EvaluationContext executionContext)
                        {
                            if (facts.Count == 0 || keyIndices.Count == 0)
                            {
                                return 0;
                            }

                            const int MaxSampleRows = 128;
                            var seenKeys = new HashSet<RowWrapper>(StructuralRowWrapperComparer);
                            var sampledRows = 0;
                            var candidates = pattern is null
                                ? facts
                                : SelectFactsForPattern(predicate, facts, pattern, executionContext);

                            foreach (var tuple in candidates)
                            {
                                if (tuple is null)
                                {
                                    continue;
                                }

                                if (pattern is not null && !TupleMatchesPattern(tuple, pattern))
                                {
                                    continue;
                                }

                                if (filter is not null && !filter(tuple))
                                {
                                    continue;
                                }

                                var key = new object[keyIndices.Count];
                                for (var i = 0; i < keyIndices.Count; i++)
                                {
                                    key[i] = GetLookupKey(tuple, keyIndices[i], NullFactIndexKey);
                                }

                                seenKeys.Add(new RowWrapper(key));
                                sampledRows++;
                                if (sampledRows >= MaxSampleRows)
                                {
                                    break;
                                }
                            }

                            return seenKeys.Count;
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

                        var leftScanBuildCost = (long)leftFacts.Count + estimatedRightProbe;
                        var rightScanBuildCost = (long)rightFacts.Count + estimatedLeftProbe;
                        var hasScanBuildCostTie = leftScanBuildCost == rightScanBuildCost;
                        var buildScanOnLeft = leftScanBuildCost <= rightScanBuildCost;

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
                        else if (hasScanBuildCostTie && keyCount > 1)
                        {
                            var leftDistinctKeyCount = EstimateDistinctScanJoinKeyCount(
                                leftScanPredicate,
                                leftFacts,
                                leftScanPattern,
                                leftScanFilter,
                                join.LeftKeys,
                                context);
                            var rightDistinctKeyCount = EstimateDistinctScanJoinKeyCount(
                                rightScanPredicate,
                                rightFacts,
                                rightScanPattern,
                                rightScanFilter,
                                join.RightKeys,
                                context);

                            if (leftDistinctKeyCount != rightDistinctKeyCount)
                            {
                                buildScanOnLeft = leftDistinctKeyCount >= rightDistinctKeyCount;
                                trace?.RecordStrategy(join, "KeyJoinScanBuildDistinctTieBreak");
                                trace?.RecordStrategy(join, buildScanOnLeft
                                    ? "KeyJoinScanBuildDistinctTieBreakLeft"
                                    : "KeyJoinScanBuildDistinctTieBreakRight");
                            }
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
                        var facts = GetScanFactsList(rightScanPredicate, context, join);
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
                        var facts = GetScanFactsList(leftScanPredicate, context, join);
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

            static int EstimateDistinctJoinKeyCount(IEnumerable<object[]> source, IReadOnlyList<int> keyIndices)
            {
                if (keyIndices.Count == 0)
                {
                    return 0;
                }

                const int MaxSampleRows = 128;
                var seenKeys = new HashSet<RowWrapper>(StructuralRowWrapperComparer);
                var sampledRows = 0;

                foreach (var tuple in source)
                {
                    if (tuple is null)
                    {
                        continue;
                    }

                    var key = new object[keyIndices.Count];
                    for (var i = 0; i < keyIndices.Count; i++)
                    {
                        key[i] = GetLookupKey(tuple, keyIndices[i], NullFactIndexKey);
                    }

                    seenKeys.Add(new RowWrapper(key));
                    sampledRows++;
                    if (sampledRows >= MaxSampleRows)
                    {
                        break;
                    }
                }

                return seenKeys.Count;
            }

            var buildLeft = EstimateBuildCost(join.Left) < EstimateBuildCost(join.Right);
            var hasRowUpperBoundTie = false;
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
                    else
                    {
                        hasRowUpperBoundTie = true;
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

            static bool HasCachedRecursiveFactIndex(
                EvaluationContext context,
                PredicateId predicate,
                RecursiveRefKind kind,
                IReadOnlyList<int> keyIndices,
                IReadOnlyList<object[]> rows)
            {
                foreach (var keyIndex in keyIndices)
                {
                    if (context.RecursiveFactIndices.TryGetValue((predicate, kind, keyIndex), out var cache) &&
                        ReferenceEquals(cache.Rows, rows))
                    {
                        return true;
                    }
                }

                return false;
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

            if (context is not null &&
                joinKeyCount > 1 &&
                hasRowUpperBoundTie &&
                !leftIsRecursive &&
                !rightIsRecursive)
            {
                var leftDistinctKeyCount = EstimateDistinctJoinKeyCount(Evaluate(join.Left, context), join.LeftKeys);
                var rightDistinctKeyCount = EstimateDistinctJoinKeyCount(Evaluate(join.Right, context), join.RightKeys);
                if (leftDistinctKeyCount != rightDistinctKeyCount)
                {
                    buildLeft = leftDistinctKeyCount >= rightDistinctKeyCount;
                    trace?.RecordStrategy(join, "KeyJoinBuildDistinctTieBreak");
                    trace?.RecordStrategy(join, buildLeft
                        ? "KeyJoinBuildDistinctTieBreakLeft"
                        : "KeyJoinBuildDistinctTieBreakRight");
                }
            }

            if (context is not null && (leftIsRecursive || rightIsRecursive))
            {
                if (leftIsRecursive && rightIsRecursive)
                {
                    var usedRecursiveBuildSelectivity = false;
                    if (joinKeyCount > 1)
                    {
                        if (leftRecursiveFilter is not null || rightRecursiveFilter is not null)
                        {
                            var leftFilteredRowCount = EstimateFilteredRecursiveRowCount(leftRecursiveRows, leftRecursiveFilter);
                            var rightFilteredRowCount = EstimateFilteredRecursiveRowCount(rightRecursiveRows, rightRecursiveFilter);

                            if (leftFilteredRowCount != rightFilteredRowCount)
                            {
                                // When both recursive sides are filtered and highly skewed, prefer the larger side as
                                // the index/build side so the smaller side can be treated as a tiny probe.
                                var smallerFilteredRowCount = Math.Min(leftFilteredRowCount, rightFilteredRowCount);
                                var largerFilteredRowCount = Math.Max(leftFilteredRowCount, rightFilteredRowCount);
                                var dualFilteredTinyProbeCandidate =
                                    leftRecursiveFilter is not null &&
                                    rightRecursiveFilter is not null &&
                                    smallerFilteredRowCount > 0 &&
                                    smallerFilteredRowCount <= 32 &&
                                    largerFilteredRowCount >= smallerFilteredRowCount * 2;

                                if (dualFilteredTinyProbeCandidate)
                                {
                                    buildLeft = leftFilteredRowCount >= rightFilteredRowCount;
                                    trace?.RecordStrategy(join, buildLeft
                                        ? "KeyJoinRecursiveBuildFilteredTinyProbeCandidateRight"
                                        : "KeyJoinRecursiveBuildFilteredTinyProbeCandidateLeft");
                                    usedRecursiveBuildSelectivity = true;
                                }
                                else
                                {
                                    buildLeft = leftFilteredRowCount <= rightFilteredRowCount;
                                    trace?.RecordStrategy(join, buildLeft ? "KeyJoinRecursiveBuildFilteredLeft" : "KeyJoinRecursiveBuildFilteredRight");
                                    usedRecursiveBuildSelectivity = true;
                                }
                            }
                            else if (leftFilteredRowCount > 0)
                            {
                                var leftDistinctKeyCount = EstimateDistinctRecursiveJoinKeyCount(leftRecursiveRows, leftRecursiveFilter, join.LeftKeys);
                                var rightDistinctKeyCount = EstimateDistinctRecursiveJoinKeyCount(rightRecursiveRows, rightRecursiveFilter, join.RightKeys);

                                if (leftDistinctKeyCount != rightDistinctKeyCount)
                                {
                                    buildLeft = leftDistinctKeyCount >= rightDistinctKeyCount;
                                    trace?.RecordStrategy(join, "KeyJoinRecursiveBuildFilteredDistinct");
                                    trace?.RecordStrategy(join, buildLeft
                                        ? "KeyJoinRecursiveBuildFilteredDistinctLeft"
                                        : "KeyJoinRecursiveBuildFilteredDistinctRight");
                                    usedRecursiveBuildSelectivity = true;
                                }
                            }
                        }
                        else
                        {
                            var leftDistinctKeyCount = EstimateDistinctRecursiveJoinKeyCount(leftRecursiveRows, null, join.LeftKeys);
                            var rightDistinctKeyCount = EstimateDistinctRecursiveJoinKeyCount(rightRecursiveRows, null, join.RightKeys);

                            if (leftDistinctKeyCount != rightDistinctKeyCount)
                            {
                                buildLeft = leftDistinctKeyCount >= rightDistinctKeyCount;
                                trace?.RecordStrategy(join, "KeyJoinRecursiveBuildDistinct");
                                trace?.RecordStrategy(join, buildLeft
                                    ? "KeyJoinRecursiveBuildDistinctLeft"
                                    : "KeyJoinRecursiveBuildDistinctRight");
                                usedRecursiveBuildSelectivity = true;
                            }
                        }
                    }

                    if (!usedRecursiveBuildSelectivity)
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

                var estimatedLeftBuildRows = EstimateFilteredRecursiveRowCount(leftRecursiveRows, leftRecursiveFilter);
                int? estimatedRightProbeUpperBound = null;
                if (TryEstimateRowUpperBound(join.Right, context, out var rightProbeUpperBoundEstimate) &&
                    rightProbeUpperBoundEstimate != int.MaxValue)
                {
                    estimatedRightProbeUpperBound = rightProbeUpperBoundEstimate;
                }

                var tinyProbeUpperBound = ComputeAdaptiveTinyProbeUpperBound(
                    defaultUpperBound: 64,
                    minUpperBound: 16,
                    maxUpperBound: 256,
                    estimatedBuildRows: estimatedLeftBuildRows,
                    estimatedProbeUpperBound: estimatedRightProbeUpperBound);
                var tinyProbeHashBuildUpperBound = Math.Clamp(Math.Max(32, tinyProbeUpperBound / 2), 16, 128);

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
                        if (probeRows.Count > tinyProbeUpperBound)
                        {
                            probeTooLarge = true;
                            break;
                        }
                    }

                    if (!probeTooLarge)
                    {
                        var dualFilteredTinyProbe =
                            leftRecursiveFilter is not null &&
                            rightRecursiveFilter is not null &&
                            estimatedLeftBuildRows >= Math.Max(8, probeRows.Count * 2);
                        if (probeRows.Count > 0 &&
                            probeRows.Count <= tinyProbeHashBuildUpperBound &&
                            ((leftRecursiveFilter is null && rightRecursiveFilter is not null) || dualFilteredTinyProbe) &&
                            !HasCachedRecursiveFactIndex(context, leftPredicate, leftKind, join.LeftKeys, leftRecursiveRows))
                        {
                            trace?.RecordStrategy(join, "KeyJoinRecursiveTinyProbeHashBuildRight");

                            var tinyProbeIndex = new Dictionary<RowWrapper, List<object[]>>(new RowWrapperComparer(StructuralArrayComparer.Instance));
                            foreach (var probeRow in probeRows)
                            {
                                var key = BuildKeyFromTuple(probeRow.Tuple, join.RightKeys);
                                var wrapper = new RowWrapper(key);
                                if (!tinyProbeIndex.TryGetValue(wrapper, out var bucket))
                                {
                                    bucket = new List<object[]>();
                                    tinyProbeIndex[wrapper] = bucket;
                                }

                                bucket.Add(probeRow.Tuple);
                            }

                            foreach (var leftTuple in leftRecursiveRows)
                            {
                                if (leftTuple is null) continue;
                                if (leftRecursiveFilter is not null && !leftRecursiveFilter(leftTuple)) continue;

                                var key = BuildKeyFromTuple(leftTuple, join.LeftKeys);
                                var wrapper = new RowWrapper(key);
                                if (!tinyProbeIndex.TryGetValue(wrapper, out var bucket))
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

                var estimatedRightBuildRows = EstimateFilteredRecursiveRowCount(rightRecursiveRows, rightRecursiveFilter);
                int? estimatedLeftProbeUpperBound = null;
                if (TryEstimateRowUpperBound(join.Left, context, out var leftProbeUpperBoundEstimate) &&
                    leftProbeUpperBoundEstimate != int.MaxValue)
                {
                    estimatedLeftProbeUpperBound = leftProbeUpperBoundEstimate;
                }

                var tinyProbeUpperBound = ComputeAdaptiveTinyProbeUpperBound(
                    defaultUpperBound: 64,
                    minUpperBound: 16,
                    maxUpperBound: 256,
                    estimatedBuildRows: estimatedRightBuildRows,
                    estimatedProbeUpperBound: estimatedLeftProbeUpperBound);
                var tinyProbeHashBuildUpperBound = Math.Clamp(Math.Max(32, tinyProbeUpperBound / 2), 16, 128);

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
                        if (probeRows.Count > tinyProbeUpperBound)
                        {
                            probeTooLarge = true;
                            break;
                        }
                    }

                    if (!probeTooLarge)
                    {
                        var dualFilteredTinyProbe =
                            rightRecursiveFilter is not null &&
                            leftRecursiveFilter is not null &&
                            estimatedRightBuildRows >= Math.Max(8, probeRows.Count * 2);
                        if (probeRows.Count > 0 &&
                            probeRows.Count <= tinyProbeHashBuildUpperBound &&
                            ((rightRecursiveFilter is null && leftRecursiveFilter is not null) || dualFilteredTinyProbe) &&
                            !HasCachedRecursiveFactIndex(context, rightPredicate, rightKind, join.RightKeys, rightRecursiveRows))
                        {
                            trace?.RecordStrategy(join, "KeyJoinRecursiveTinyProbeHashBuildLeft");

                            var tinyProbeIndex = new Dictionary<RowWrapper, List<object[]>>(new RowWrapperComparer(StructuralArrayComparer.Instance));
                            foreach (var probeRow in probeRows)
                            {
                                var key = BuildKeyFromTuple(probeRow.Tuple, join.LeftKeys);
                                var wrapper = new RowWrapper(key);
                                if (!tinyProbeIndex.TryGetValue(wrapper, out var bucket))
                                {
                                    bucket = new List<object[]>();
                                    tinyProbeIndex[wrapper] = bucket;
                                }

                                bucket.Add(probeRow.Tuple);
                            }

                            foreach (var rightTuple in rightRecursiveRows)
                            {
                                if (rightTuple is null) continue;
                                if (rightRecursiveFilter is not null && !rightRecursiveFilter(rightTuple)) continue;

                                var key = BuildKeyFromTuple(rightTuple, join.RightKeys);
                                var wrapper = new RowWrapper(key);
                                if (!tinyProbeIndex.TryGetValue(wrapper, out var bucket))
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
                    var facts = GetScanFactsList(leftScan.Relation, context, join);
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
                    var facts = GetScanFactsList(rightScan.Relation, context, join);
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

        private bool TryBindRelation(PredicateId predicate, RelationRetentionMode preferredMode, out RelationBinding binding)
        {
            if (_provider is IRetentionAwareRelationProvider retentionAwareProvider &&
                retentionAwareProvider.TryBindRelation(predicate, preferredMode, out binding))
            {
                return true;
            }

            binding = default;
            return false;
        }

        private bool TryGetDelimitedSource(PredicateId predicate, RelationRetentionMode preferredMode, out DelimitedRelationSource source)
        {
            if (TryBindRelation(predicate, preferredMode, out var binding) && binding.DelimitedSource is { } delimitedSource)
            {
                source = delimitedSource;
                return true;
            }

            source = default;
            return false;
        }

        private bool TryGetReplayableSource(PredicateId predicate, EvaluationContext? context, out IReplayableRelationSource source)
        {
            if (context is not null && context.ReplayableFactSources.TryGetValue(predicate, out var cached))
            {
                source = cached;
                return true;
            }

            if (TryBindRelation(predicate, RelationRetentionMode.Replayable, out var binding) &&
                binding.ReplayableSource is { } replayableSource)
            {
                if (context is not null)
                {
                    context.ReplayableFactSources[predicate] = replayableSource;
                }

                source = replayableSource;
                return true;
            }

            source = default!;
            return false;
        }

        private bool TryGetExternalFacts(PredicateId predicate, out IEnumerable<object[]> facts)
        {
            if (TryBindRelation(predicate, RelationRetentionMode.ExternalMaterialized, out var binding) &&
                binding.Mode == RelationRetentionMode.ExternalMaterialized)
            {
                facts = _provider.GetFacts(predicate) ?? Enumerable.Empty<object[]>();
                return true;
            }

            facts = default!;
            return false;
        }

        private static void AddPathAwareEdge(
            Dictionary<object, PathAwareSuccessorBucket> successors,
            List<object?> seeds,
            HashSet<object?> seenSeeds,
            object? source,
            object? target)
        {
            var lookupKey = source ?? NullFactIndexKey;
            if (!successors.TryGetValue(lookupKey, out var bucket))
            {
                bucket = new PathAwareSuccessorBucket(source);
                successors[lookupKey] = bucket;
            }

            bucket.Targets.Add(target);
            if (seenSeeds.Add(source))
            {
                seeds.Add(source);
            }
        }

        private static PathAwareEdgeState FinalizePathAwareEdgeState(
            Dictionary<object, PathAwareSuccessorBucket> successors,
            List<object?> seeds)
        {
            seeds.Sort(CompareCacheSeedValues);
            return new PathAwareEdgeState(successors, seeds);
        }

        private static PathAwareEdgeState BuildPathAwareEdgeStateFromRows(IEnumerable<object[]> edges)
        {
            var successors = new Dictionary<object, PathAwareSuccessorBucket>();
            var seeds = new List<object?>();
            var seenSeeds = new HashSet<object?>();

            foreach (var edge in edges)
            {
                if (edge is null || edge.Length < 2)
                {
                    continue;
                }

                AddPathAwareEdge(successors, seeds, seenSeeds, edge[0], edge[1]);
            }

            return FinalizePathAwareEdgeState(successors, seeds);
        }

        private static PathAwareEdgeState BuildPathAwareEdgeStateFromDelimited(DelimitedRelationSource delimitedSource)
        {
            var successors = new Dictionary<object, PathAwareSuccessorBucket>();
            var seeds = new List<object?>();
            var seenSeeds = new HashSet<object?>();

            using var reader = DelimitedRelationReader.OpenSequentialReader(delimitedSource.InputPath);
            for (var i = 0; i < delimitedSource.SkipRows; i++)
            {
                if (reader.ReadLine() is null)
                {
                    break;
                }
            }

            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                if (!DelimitedRelationReader.TrySplitTwoColumnLine(line, delimitedSource.Delimiter, out var left, out var right))
                {
                    continue;
                }

                AddPathAwareEdge(successors, seeds, seenSeeds, left, right);
            }

            return FinalizePathAwareEdgeState(successors, seeds);
        }

        private static void ProbePathAwareEdgeRows(IEnumerable<object[]> edges, int maxRows)
        {
            var successors = new Dictionary<object, PathAwareSuccessorBucket>();
            var seeds = new List<object?>();
            var seenSeeds = new HashSet<object?>();
            var consumed = 0;

            foreach (var edge in edges)
            {
                if (edge is null || edge.Length < 2)
                {
                    continue;
                }

                AddPathAwareEdge(successors, seeds, seenSeeds, edge[0], edge[1]);
                consumed++;
                if (consumed >= maxRows)
                {
                    break;
                }
            }

            seeds.Sort(CompareCacheSeedValues);
        }

        private static void ProbePathAwareEdgeDelimited(DelimitedRelationSource delimitedSource, int maxRows)
        {
            var successors = new Dictionary<object, PathAwareSuccessorBucket>();
            var seeds = new List<object?>();
            var seenSeeds = new HashSet<object?>();
            var consumed = 0;

            using var reader = DelimitedRelationReader.OpenSequentialReader(delimitedSource.InputPath);
            for (var i = 0; i < delimitedSource.SkipRows; i++)
            {
                if (reader.ReadLine() is null)
                {
                    break;
                }
            }

            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                if (!DelimitedRelationReader.TrySplitTwoColumnLine(line, delimitedSource.Delimiter, out var left, out var right))
                {
                    continue;
                }

                AddPathAwareEdge(successors, seeds, seenSeeds, left, right);
                consumed++;
                if (consumed >= maxRows)
                {
                    break;
                }
            }

            seeds.Sort(CompareCacheSeedValues);
        }

        private static void ProbeDagRows(IEnumerable<object[]> edges, IEnumerable<object[]> seeds, int maxEdgeRows, int maxSeedRows)
        {
            var nodeIds = new Dictionary<object?, int>();
            var edgeCount = 0;

            foreach (var edge in edges)
            {
                if (edge is null || edge.Length < 2)
                {
                    continue;
                }

                if (!nodeIds.ContainsKey(edge[0]))
                {
                    nodeIds.Add(edge[0], nodeIds.Count);
                }

                if (!nodeIds.ContainsKey(edge[1]))
                {
                    nodeIds.Add(edge[1], nodeIds.Count);
                }

                edgeCount++;
                if (edgeCount >= maxEdgeRows)
                {
                    break;
                }
            }

            var groupIds = new Dictionary<object?, int>();
            var seedCount = 0;
            foreach (var seed in seeds)
            {
                if (seed is null || seed.Length < 2)
                {
                    continue;
                }

                if (!nodeIds.ContainsKey(seed[1]))
                {
                    continue;
                }

                if (!groupIds.ContainsKey(seed[0]))
                {
                    groupIds.Add(seed[0], groupIds.Count);
                }

                seedCount++;
                if (seedCount >= maxSeedRows)
                {
                    break;
                }
            }
        }

        private static void ProbeDagDelimited(DelimitedRelationSource edgeSource, DelimitedRelationSource seedSource, int maxEdgeRows, int maxSeedRows)
        {
            var nodeIds = new Dictionary<string, int>(StringComparer.Ordinal);
            var edgeCount = 0;

            using (var reader = DelimitedRelationReader.OpenSequentialReader(edgeSource.InputPath))
            {
                for (var i = 0; i < edgeSource.SkipRows; i++)
                {
                    if (reader.ReadLine() is null)
                    {
                        return;
                    }
                }

                string? line;
                while ((line = reader.ReadLine()) is not null)
                {
                    if (!DelimitedRelationReader.TrySplitTwoColumnLine(line, edgeSource.Delimiter, out var left, out var right))
                    {
                        continue;
                    }

                    if (!nodeIds.ContainsKey(left))
                    {
                        nodeIds.Add(left, nodeIds.Count);
                    }

                    if (!nodeIds.ContainsKey(right))
                    {
                        nodeIds.Add(right, nodeIds.Count);
                    }

                    edgeCount++;
                    if (edgeCount >= maxEdgeRows)
                    {
                        break;
                    }
                }
            }

            var groupIds = new Dictionary<string, int>(StringComparer.Ordinal);
            var seedCount = 0;
            using (var reader = DelimitedRelationReader.OpenSequentialReader(seedSource.InputPath))
            {
                for (var i = 0; i < seedSource.SkipRows; i++)
                {
                    if (reader.ReadLine() is null)
                    {
                        return;
                    }
                }

                string? line;
                while ((line = reader.ReadLine()) is not null)
                {
                    if (!DelimitedRelationReader.TrySplitTwoColumnLine(line, seedSource.Delimiter, out var group, out var node))
                    {
                        continue;
                    }

                    if (!nodeIds.ContainsKey(node))
                    {
                        continue;
                    }

                    if (!groupIds.ContainsKey(group))
                    {
                        groupIds.Add(group, groupIds.Count);
                    }

                    seedCount++;
                    if (seedCount >= maxSeedRows)
                    {
                        break;
                    }
                }
            }
        }

        private static void ProbeFactRows(IEnumerable<object[]> facts, int maxRows)
        {
            var consumed = 0;
            foreach (var fact in facts)
            {
                if (fact is null)
                {
                    continue;
                }

                consumed++;
                if (consumed >= maxRows)
                {
                    break;
                }
            }
        }

        private ScanRelationRetentionSelection GetScanRelationRetentionSelection(
            PredicateId predicate,
            ScanRelationAccessKind accessKind,
            EvaluationContext context,
            PlanNode traceNode)
        {
            var cacheKey = (predicate, accessKind);
            if (context.ScanRelationRetentionSelections.TryGetValue(cacheKey, out var cachedSelection))
            {
                return cachedSelection;
            }

            var hasStreaming = TryGetDelimitedSource(predicate, RelationRetentionMode.Streaming, out var delimitedSource);
            var hasReplayable = TryGetReplayableSource(predicate, context, out var replayableSource);
            var hasExternal = TryGetExternalFacts(predicate, out var externalFacts);
            var concreteReplayable = replayableSource as ReplayableRelationSource;
            var relationBytes = hasStreaming && !string.IsNullOrEmpty(delimitedSource.InputPath) && File.Exists(delimitedSource.InputPath)
                ? new FileInfo(delimitedSource.InputPath).Length
                : (long?)null;
            const int scanProbeRowLimit = 256;

            Func<TimeSpan>? measureStreamingProbe = hasStreaming
                ? () => MeasureElapsed(() => ProbeFactRows(DelimitedRelationReader.ReadRows(delimitedSource), scanProbeRowLimit))
                : null;
            Func<TimeSpan>? measureReplayableProbe = hasReplayable
                ? () => MeasureElapsed(() =>
                {
                    var probeRows = concreteReplayable is not null
                        ? concreteReplayable.ProbeMaterialize(scanProbeRowLimit)
                        : replayableSource.Stream().Take(scanProbeRowLimit).ToList();
                    ProbeFactRows(probeRows, scanProbeRowLimit);
                })
                : null;
            Func<TimeSpan>? measureExternalProbe = hasExternal
                ? () => MeasureElapsed(() => ProbeFactRows(externalFacts, scanProbeRowLimit))
                : null;

            var selection = ResolveScanRelationRetentionStrategy(
                context.Trace,
                traceNode,
                _scanRelationRetentionStrategy,
                accessKind,
                relationBytes,
                hasStreaming,
                hasReplayable,
                hasExternal,
                concreteReplayable?.IsMaterialized == true,
                measureStreamingProbe,
                measureReplayableProbe,
                measureExternalProbe);
            context.ScanRelationRetentionSelections[cacheKey] = selection;
            return selection;
        }

        private MaterializationPlanSelection ResolveScanMaterializationPlan(
            PredicateId predicate,
            ScanRelationAccessKind accessKind,
            EvaluationContext context,
            PlanNode traceNode)
        {
            var relationRetentionSelection = ToRelationRetentionSelection(
                GetScanRelationRetentionSelection(predicate, accessKind, context, traceNode));
            var plan = ResolveMaterializationPlan(context.Trace, traceNode, relationRetentionSelection);
            RecordScanRelationRetentionStrategy(
                context.Trace,
                traceNode,
                new ScanRelationRetentionSelection(ToScanRelationRetentionStrategy(plan.RelationRetention.Strategy), plan.RelationRetention.DecisionMode));
            RecordMaterializationPlanStrategy(context.Trace, traceNode, "Scan", plan);
            return plan;
        }

        private IEnumerable<object[]> GetScanFactStream(PredicateId predicate, EvaluationContext? context, PlanNode traceNode)
        {
            if (context is null)
            {
                return GetFactStream(predicate, context);
            }

            if (context.Facts.TryGetValue(predicate, out var cachedFacts))
            {
                context.Trace?.RecordCacheLookup("Facts", $"{predicate.Name}/{predicate.Arity}", hit: true, built: false);
                return cachedFacts;
            }

            var plan = ResolveScanMaterializationPlan(predicate, ScanRelationAccessKind.Stream, context, traceNode);
            var hasStreaming = TryGetDelimitedSource(predicate, RelationRetentionMode.Streaming, out var delimitedSource);
            var hasReplayable = TryGetReplayableSource(predicate, context, out var replayableSource);
            var hasExternal = TryGetExternalFacts(predicate, out var externalFacts);

            return plan.RelationRetention.Strategy switch
            {
                RelationRetentionPolicyStrategy.StreamingDirect when hasStreaming => DelimitedRelationReader.ReadRows(delimitedSource),
                RelationRetentionPolicyStrategy.ReplayableBuffer when hasReplayable => replayableSource.Stream(),
                RelationRetentionPolicyStrategy.ExternalMaterialized when hasExternal => externalFacts,
                _ when hasReplayable => replayableSource.Stream(),
                _ when hasStreaming => DelimitedRelationReader.ReadRows(delimitedSource),
                _ => _provider.GetFacts(predicate) ?? Enumerable.Empty<object[]>()
            };
        }

        private List<object[]> GetScanFactsList(PredicateId predicate, EvaluationContext? context, PlanNode traceNode)
        {
            if (context is null)
            {
                return MaterializeFacts(predicate, context);
            }

            if (context.Facts.TryGetValue(predicate, out var cached))
            {
                context.Trace?.RecordCacheLookup("Facts", $"{predicate.Name}/{predicate.Arity}", hit: true, built: false);
                return cached;
            }

            context.Trace?.RecordCacheLookup("Facts", $"{predicate.Name}/{predicate.Arity}", hit: false, built: true);

            var plan = ResolveScanMaterializationPlan(predicate, ScanRelationAccessKind.List, context, traceNode);
            var hasStreaming = TryGetDelimitedSource(predicate, RelationRetentionMode.Streaming, out var delimitedSource);
            var hasReplayable = TryGetReplayableSource(predicate, context, out var replayableSource);
            var hasExternal = TryGetExternalFacts(predicate, out var externalFacts);
            List<object[]> facts;

            switch (plan.RelationRetention.Strategy)
            {
                case RelationRetentionPolicyStrategy.StreamingDirect when hasStreaming:
                    facts = MeasurePhase(context.Trace, traceNode, "scan_materialize_streaming_direct", () => DelimitedRelationReader.ReadRows(delimitedSource).ToList());
                    break;
                case RelationRetentionPolicyStrategy.ReplayableBuffer when hasReplayable:
                    facts = MeasurePhase(context.Trace, traceNode, "scan_materialize_replayable", () => replayableSource.Materialize());
                    break;
                case RelationRetentionPolicyStrategy.ExternalMaterialized when hasExternal:
                    facts = MeasurePhase(context.Trace, traceNode, "scan_materialize_external_materialized", () => externalFacts as List<object[]> ?? externalFacts.ToList());
                    break;
                default:
                    if (hasReplayable)
                    {
                        facts = MeasurePhase(context.Trace, traceNode, "scan_materialize_replayable", () => replayableSource.Materialize());
                    }
                    else if (hasStreaming)
                    {
                        facts = MeasurePhase(context.Trace, traceNode, "scan_materialize_streaming_direct", () => DelimitedRelationReader.ReadRows(delimitedSource).ToList());
                    }
                    else
                    {
                        var externalSource = _provider.GetFacts(predicate) ?? Enumerable.Empty<object[]>();
                        facts = MeasurePhase(context.Trace, traceNode, "scan_materialize_external_materialized", () => externalSource as List<object[]> ?? externalSource.ToList());
                    }
                    break;
            }

            context.Facts[predicate] = facts;
            return facts;
        }

        private HashSet<object[]> GetScanFactsSet(PredicateId predicate, EvaluationContext? context, PlanNode traceNode)
        {
            if (context is null)
            {
                return new HashSet<object[]>(MaterializeFacts(predicate, context), StructuralArrayComparer.Instance);
            }

            if (context.FactSets.TryGetValue(predicate, out var cached))
            {
                context.Trace?.RecordCacheLookup("FactSet", $"{predicate.Name}/{predicate.Arity}", hit: true, built: false);
                return cached;
            }

            context.Trace?.RecordCacheLookup("FactSet", $"{predicate.Name}/{predicate.Arity}", hit: false, built: true);
            var facts = GetScanFactsList(predicate, context, traceNode);
            var set = MeasurePhase(context.Trace, traceNode, "scan_build_fact_set", () => new HashSet<object[]>(facts, StructuralArrayComparer.Instance));
            context.FactSets[predicate] = set;
            return set;
        }

        private ClosureRelationRetentionSelection GetClosureRelationRetentionSelection(
            PredicateId predicate,
            ClosureRelationAccessKind accessKind,
            EvaluationContext context,
            PlanNode traceNode)
        {
            var cacheKey = (predicate, accessKind);
            if (context.ClosureRelationRetentionSelections.TryGetValue(cacheKey, out var cachedSelection))
            {
                return cachedSelection;
            }

            var hasStreaming = TryGetDelimitedSource(predicate, RelationRetentionMode.Streaming, out var delimitedSource);
            var hasReplayable = TryGetReplayableSource(predicate, context, out var replayableSource);
            var hasExternal = TryGetExternalFacts(predicate, out var externalFacts);
            var concreteReplayable = replayableSource as ReplayableRelationSource;
            var relationBytes = hasStreaming && !string.IsNullOrEmpty(delimitedSource.InputPath) && File.Exists(delimitedSource.InputPath)
                ? new FileInfo(delimitedSource.InputPath).Length
                : (long?)null;
            const int closureProbeRowLimit = 256;

            Func<TimeSpan>? measureStreamingProbe = hasStreaming
                ? () => MeasureElapsed(() => ProbeFactRows(DelimitedRelationReader.ReadRows(delimitedSource), closureProbeRowLimit))
                : null;
            Func<TimeSpan>? measureReplayableProbe = hasReplayable
                ? () => MeasureElapsed(() =>
                {
                    var probeRows = concreteReplayable is not null
                        ? concreteReplayable.ProbeMaterialize(closureProbeRowLimit)
                        : replayableSource.Stream().Take(closureProbeRowLimit).ToList();
                    ProbeFactRows(probeRows, closureProbeRowLimit);
                })
                : null;
            Func<TimeSpan>? measureExternalProbe = hasExternal
                ? () => MeasureElapsed(() => ProbeFactRows(externalFacts, closureProbeRowLimit))
                : null;

            var selection = ResolveClosureRelationRetentionStrategy(
                context.Trace,
                traceNode,
                _closureRelationRetentionStrategy,
                accessKind,
                relationBytes,
                hasStreaming,
                hasReplayable,
                hasExternal,
                concreteReplayable?.IsMaterialized == true,
                measureStreamingProbe,
                measureReplayableProbe,
                measureExternalProbe);
            context.ClosureRelationRetentionSelections[cacheKey] = selection;
            return selection;
        }

        private List<object[]> GetClosureFactsList(
            PredicateId predicate,
            ClosureRelationAccessKind accessKind,
            EvaluationContext? context,
            PlanNode traceNode)
        {
            if (context is null)
            {
                return MaterializeFacts(predicate, context);
            }

            if (context.Facts.TryGetValue(predicate, out var cached))
            {
                context.Trace?.RecordCacheLookup("Facts", $"{predicate.Name}/{predicate.Arity}", hit: true, built: false);
                return cached;
            }

            context.Trace?.RecordCacheLookup("Facts", $"{predicate.Name}/{predicate.Arity}", hit: false, built: true);

            var plan = ResolveMaterializationPlan(
                context.Trace,
                traceNode,
                ToRelationRetentionSelection(GetClosureRelationRetentionSelection(predicate, accessKind, context, traceNode)));
            RecordClosureRelationRetentionStrategy(
                context.Trace,
                traceNode,
                new ClosureRelationRetentionSelection(ToClosureRelationRetentionStrategy(plan.RelationRetention.Strategy), plan.RelationRetention.DecisionMode));
            RecordMaterializationPlanStrategy(context.Trace, traceNode, "Closure", plan);

            var hasStreaming = TryGetDelimitedSource(predicate, RelationRetentionMode.Streaming, out var delimitedSource);
            var hasReplayable = TryGetReplayableSource(predicate, context, out var replayableSource);
            var hasExternal = TryGetExternalFacts(predicate, out var externalFacts);
            List<object[]> facts;

            switch (plan.RelationRetention.Strategy)
            {
                case RelationRetentionPolicyStrategy.StreamingDirect when hasStreaming:
                    facts = MeasurePhase(context.Trace, traceNode, "closure_materialize_streaming_direct", () => DelimitedRelationReader.ReadRows(delimitedSource).ToList());
                    break;
                case RelationRetentionPolicyStrategy.ReplayableBuffer when hasReplayable:
                    facts = MeasurePhase(context.Trace, traceNode, "closure_materialize_replayable", () => replayableSource.Materialize());
                    break;
                case RelationRetentionPolicyStrategy.ExternalMaterialized when hasExternal:
                    facts = MeasurePhase(context.Trace, traceNode, "closure_materialize_external_materialized", () => externalFacts as List<object[]> ?? externalFacts.ToList());
                    break;
                default:
                    if (hasReplayable)
                    {
                        facts = MeasurePhase(context.Trace, traceNode, "closure_materialize_replayable", () => replayableSource.Materialize());
                    }
                    else if (hasStreaming)
                    {
                        facts = MeasurePhase(context.Trace, traceNode, "closure_materialize_streaming_direct", () => DelimitedRelationReader.ReadRows(delimitedSource).ToList());
                    }
                    else
                    {
                        var externalSource = _provider.GetFacts(predicate) ?? Enumerable.Empty<object[]>();
                        facts = MeasurePhase(context.Trace, traceNode, "closure_materialize_external_materialized", () => externalSource as List<object[]> ?? externalSource.ToList());
                    }
                    break;
            }

            context.Facts[predicate] = facts;
            return facts;
        }

        private PathAwareSupportRelationRetentionSelection GetPathAwareSupportRelationRetentionSelection(
            PredicateId predicate,
            PathAwareSupportRelationAccessKind accessKind,
            EvaluationContext context,
            PlanNode traceNode)
        {
            var cacheKey = (predicate, accessKind);
            if (context.PathAwareSupportRelationRetentionSelections.TryGetValue(cacheKey, out var cachedSelection))
            {
                return cachedSelection;
            }

            var hasStreaming = TryGetDelimitedSource(predicate, RelationRetentionMode.Streaming, out var delimitedSource);
            var hasReplayable = TryGetReplayableSource(predicate, context, out var replayableSource);
            var hasExternal = TryGetExternalFacts(predicate, out var externalFacts);
            var concreteReplayable = replayableSource as ReplayableRelationSource;
            var relationBytes = hasStreaming && !string.IsNullOrEmpty(delimitedSource.InputPath) && File.Exists(delimitedSource.InputPath)
                ? new FileInfo(delimitedSource.InputPath).Length
                : (long?)null;
            const int supportProbeRowLimit = 256;

            Func<TimeSpan>? measureStreamingProbe = hasStreaming
                ? () => MeasureElapsed(() => ProbeFactRows(DelimitedRelationReader.ReadRows(delimitedSource), supportProbeRowLimit))
                : null;
            Func<TimeSpan>? measureReplayableProbe = hasReplayable
                ? () => MeasureElapsed(() =>
                {
                    var probeRows = concreteReplayable is not null
                        ? concreteReplayable.ProbeMaterialize(supportProbeRowLimit)
                        : replayableSource.Stream().Take(supportProbeRowLimit).ToList();
                    ProbeFactRows(probeRows, supportProbeRowLimit);
                })
                : null;
            Func<TimeSpan>? measureExternalProbe = hasExternal
                ? () => MeasureElapsed(() => ProbeFactRows(externalFacts, supportProbeRowLimit))
                : null;

            var selection = ResolvePathAwareSupportRelationRetentionStrategy(
                context.Trace,
                traceNode,
                _pathAwareSupportRelationRetentionStrategy,
                accessKind,
                relationBytes,
                hasStreaming,
                hasReplayable,
                hasExternal,
                concreteReplayable?.IsMaterialized == true,
                measureStreamingProbe,
                measureReplayableProbe,
                measureExternalProbe);
            context.PathAwareSupportRelationRetentionSelections[cacheKey] = selection;
            return selection;
        }

        private IEnumerable<object[]> GetPathAwareSupportFactStream(
            PredicateId predicate,
            PathAwareSupportRelationAccessKind accessKind,
            EvaluationContext? context,
            PlanNode traceNode)
        {
            if (context is null)
            {
                return GetFactStream(predicate, context);
            }

            if (context.Facts.TryGetValue(predicate, out var cachedFacts))
            {
                context.Trace?.RecordCacheLookup("Facts", $"{predicate.Name}/{predicate.Arity}", hit: true, built: false);
                return cachedFacts;
            }

            var plan = ResolveMaterializationPlan(
                context.Trace,
                traceNode,
                ToRelationRetentionSelection(GetPathAwareSupportRelationRetentionSelection(predicate, accessKind, context, traceNode)));
            RecordPathAwareSupportRelationRetentionStrategy(
                context.Trace,
                traceNode,
                accessKind,
                new PathAwareSupportRelationRetentionSelection(ToPathAwareSupportRelationRetentionStrategy(plan.RelationRetention.Strategy), plan.RelationRetention.DecisionMode));
            RecordMaterializationPlanStrategy(
                context.Trace,
                traceNode,
                accessKind == PathAwareSupportRelationAccessKind.Roots ? "PathAwareSupportRoots" : "PathAwareSupportSeeds",
                plan);

            var hasStreaming = TryGetDelimitedSource(predicate, RelationRetentionMode.Streaming, out var delimitedSource);
            var hasReplayable = TryGetReplayableSource(predicate, context, out var replayableSource);
            var hasExternal = TryGetExternalFacts(predicate, out var externalFacts);
            var phasePrefix = accessKind == PathAwareSupportRelationAccessKind.Roots ? "support_roots" : "support_seeds";

            switch (plan.RelationRetention.Strategy)
            {
                case RelationRetentionPolicyStrategy.StreamingDirect when hasStreaming:
                    return DelimitedRelationReader.ReadRows(delimitedSource);
                case RelationRetentionPolicyStrategy.ReplayableBuffer when hasReplayable:
                {
                    var facts = MeasurePhase(context.Trace, traceNode, $"{phasePrefix}_materialize_replayable", () => replayableSource.Materialize());
                    context.Facts[predicate] = facts;
                    return facts;
                }
                case RelationRetentionPolicyStrategy.ExternalMaterialized when hasExternal:
                {
                    var facts = MeasurePhase(context.Trace, traceNode, $"{phasePrefix}_materialize_external_materialized", () => externalFacts as List<object[]> ?? externalFacts.ToList());
                    context.Facts[predicate] = facts;
                    return facts;
                }
                default:
                    if (hasReplayable)
                    {
                        var facts = MeasurePhase(context.Trace, traceNode, $"{phasePrefix}_materialize_replayable", () => replayableSource.Materialize());
                        context.Facts[predicate] = facts;
                        return facts;
                    }

                    if (hasStreaming)
                    {
                        return DelimitedRelationReader.ReadRows(delimitedSource);
                    }

                    var externalSource = _provider.GetFacts(predicate) ?? Enumerable.Empty<object[]>();
                    var externalList = MeasurePhase(context.Trace, traceNode, $"{phasePrefix}_materialize_external_materialized", () => externalSource as List<object[]> ?? externalSource.ToList());
                    context.Facts[predicate] = externalList;
                    return externalList;
            }
        }

        private List<object[]> GetDagFactsList(
            PredicateId predicate,
            DagRelationRetentionStrategy strategy,
            EvaluationContext? context,
            PlanNode traceNode,
            bool edgeRelation)
        {
            if (context is null)
            {
                return MaterializeFacts(predicate, context);
            }

            if (context.Facts.TryGetValue(predicate, out var cached))
            {
                context.Trace?.RecordCacheLookup("Facts", $"{predicate.Name}/{predicate.Arity}", hit: true, built: false);
                return cached;
            }

            context.Trace?.RecordCacheLookup("Facts", $"{predicate.Name}/{predicate.Arity}", hit: false, built: true);

            var hasStreaming = TryGetDelimitedSource(predicate, RelationRetentionMode.Streaming, out var delimitedSource);
            var hasReplayable = TryGetReplayableSource(predicate, context, out var replayableSource);
            var hasExternal = TryGetExternalFacts(predicate, out var externalFacts);
            var streamingPhase = edgeRelation ? "dag_materialize_streaming_edges" : "dag_materialize_streaming_seeds";
            var replayablePhase = edgeRelation ? "dag_materialize_replayable_edges" : "dag_materialize_replayable_seeds";
            var externalPhase = edgeRelation ? "dag_materialize_external_edges" : "dag_materialize_external_seeds";
            List<object[]> facts;

            switch (strategy)
            {
                case DagRelationRetentionStrategy.StreamingDirect when hasStreaming:
                    facts = MeasurePhase(context.Trace, traceNode, streamingPhase, () => DelimitedRelationReader.ReadRows(delimitedSource).ToList());
                    break;
                case DagRelationRetentionStrategy.ReplayableBuffer when hasReplayable:
                    facts = MeasurePhase(context.Trace, traceNode, replayablePhase, () => replayableSource.Materialize());
                    break;
                case DagRelationRetentionStrategy.ExternalMaterialized when hasExternal:
                    facts = MeasurePhase(context.Trace, traceNode, externalPhase, () => externalFacts as List<object[]> ?? externalFacts.ToList());
                    break;
                default:
                    if (hasReplayable)
                    {
                        facts = MeasurePhase(context.Trace, traceNode, replayablePhase, () => replayableSource.Materialize());
                    }
                    else if (hasStreaming)
                    {
                        facts = MeasurePhase(context.Trace, traceNode, streamingPhase, () => DelimitedRelationReader.ReadRows(delimitedSource).ToList());
                    }
                    else
                    {
                        var externalSource = _provider.GetFacts(predicate) ?? Enumerable.Empty<object[]>();
                        facts = MeasurePhase(context.Trace, traceNode, externalPhase, () => externalSource as List<object[]> ?? externalSource.ToList());
                    }
                    break;
            }

            context.Facts[predicate] = facts;
            return facts;
        }

        private IEnumerable<object[]> GetFactStream(PredicateId predicate, EvaluationContext? context = null)
        {
            if (context is not null && TryGetReplayableSource(predicate, context, out var replayableSource))
            {
                return replayableSource.Stream();
            }

            if (TryGetDelimitedSource(predicate, RelationRetentionMode.Streaming, out var source))
            {
                return DelimitedRelationReader.ReadRows(source);
            }

            if (TryGetReplayableSource(predicate, context, out replayableSource))
            {
                return replayableSource.Stream();
            }

            return _provider.GetFacts(predicate) ?? Enumerable.Empty<object[]>();
        }

        private List<object[]> MaterializeFacts(PredicateId predicate, EvaluationContext? context)
        {
            if (TryGetReplayableSource(predicate, context, out var replayableSource))
            {
                return replayableSource.Materialize();
            }

            var factsSource = _provider.GetFacts(predicate) ?? Enumerable.Empty<object[]>();
            return factsSource as List<object[]> ?? factsSource.ToList();
        }

        private PathAwareEdgeState GetPathAwareEdgeState(PredicateId predicate, EvaluationContext context, PlanNode traceNode)
        {
            if (context.PathAwareEdgeStates.TryGetValue(predicate, out var cached))
            {
                context.Trace?.RecordCacheLookup("PathAwareEdgeState", $"{predicate.Name}/{predicate.Arity}", hit: true, built: false);
                return cached;
            }

            context.Trace?.RecordCacheLookup("PathAwareEdgeState", $"{predicate.Name}/{predicate.Arity}", hit: false, built: true);

            var hasStreaming = TryGetDelimitedSource(predicate, RelationRetentionMode.Streaming, out var delimitedSource);
            var hasReplayable = TryGetReplayableSource(predicate, context, out var replayableSource);
            var hasExternal = TryGetExternalFacts(predicate, out var externalFacts);
            var concreteReplayable = replayableSource as ReplayableRelationSource;
            var relationBytes = hasStreaming && !string.IsNullOrEmpty(delimitedSource.InputPath) && File.Exists(delimitedSource.InputPath)
                ? new FileInfo(delimitedSource.InputPath).Length
                : (long?)null;
            const int edgeProbeRowLimit = 256;

            Func<TimeSpan>? measureStreamingProbe = hasStreaming
                ? () => MeasureElapsed(() => ProbePathAwareEdgeDelimited(delimitedSource, edgeProbeRowLimit))
                : null;
            Func<TimeSpan>? measureReplayableProbe = hasReplayable
                ? () => MeasureElapsed(() =>
                {
                    var probeRows = concreteReplayable is not null
                        ? concreteReplayable.ProbeMaterialize(edgeProbeRowLimit)
                        : replayableSource.Stream().Take(edgeProbeRowLimit).ToList();
                    ProbePathAwareEdgeRows(probeRows, edgeProbeRowLimit);
                })
                : null;
            Func<TimeSpan>? measureExternalProbe = hasExternal
                ? () => MeasureElapsed(() => ProbePathAwareEdgeRows(externalFacts, edgeProbeRowLimit))
                : null;

            var selection = ResolvePathAwareEdgeRetentionStrategy(
                context.Trace,
                traceNode,
                _pathAwareEdgeRetentionStrategy,
                relationBytes,
                hasStreaming,
                hasReplayable,
                hasExternal,
                concreteReplayable?.IsMaterialized == true,
                measureStreamingProbe,
                measureReplayableProbe,
                measureExternalProbe);
            RecordPathAwareEdgeRetentionStrategy(context.Trace, traceNode, selection);

            PathAwareEdgeState state;
            switch (selection.Strategy)
            {
                case PathAwareEdgeRetentionStrategy.StreamingDirect when hasStreaming:
                    state = MeasurePhase(context.Trace, traceNode, "edge_build_streaming_direct", () => BuildPathAwareEdgeStateFromDelimited(delimitedSource));
                    break;
                case PathAwareEdgeRetentionStrategy.ReplayableBuffer when hasReplayable:
                    var replayableRows = MeasurePhase(context.Trace, traceNode, "edge_materialize_replayable", () => replayableSource.Materialize());
                    state = MeasurePhase(context.Trace, traceNode, "edge_build_replayable_buffer", () => BuildPathAwareEdgeStateFromRows(replayableRows));
                    break;
                case PathAwareEdgeRetentionStrategy.ExternalMaterialized when hasExternal:
                    state = MeasurePhase(context.Trace, traceNode, "edge_build_external_materialized", () => BuildPathAwareEdgeStateFromRows(externalFacts));
                    break;
                case PathAwareEdgeRetentionStrategy.StreamingDirect:
                    if (hasReplayable)
                    {
                        var fallbackReplayableRows = MeasurePhase(context.Trace, traceNode, "edge_materialize_replayable", () => replayableSource.Materialize());
                        state = MeasurePhase(context.Trace, traceNode, "edge_build_replayable_buffer", () => BuildPathAwareEdgeStateFromRows(fallbackReplayableRows));
                        break;
                    }

                    state = MeasurePhase(context.Trace, traceNode, "edge_build_external_materialized", () => BuildPathAwareEdgeStateFromRows(_provider.GetFacts(predicate) ?? Enumerable.Empty<object[]>()));
                    break;
                case PathAwareEdgeRetentionStrategy.ReplayableBuffer:
                    if (hasStreaming)
                    {
                        state = MeasurePhase(context.Trace, traceNode, "edge_build_streaming_direct", () => BuildPathAwareEdgeStateFromDelimited(delimitedSource));
                        break;
                    }

                    state = MeasurePhase(context.Trace, traceNode, "edge_build_external_materialized", () => BuildPathAwareEdgeStateFromRows(_provider.GetFacts(predicate) ?? Enumerable.Empty<object[]>()));
                    break;
                default:
                    if (hasStreaming)
                    {
                        state = MeasurePhase(context.Trace, traceNode, "edge_build_streaming_direct", () => BuildPathAwareEdgeStateFromDelimited(delimitedSource));
                    }
                    else if (hasReplayable)
                    {
                        var fallbackReplayableRows = MeasurePhase(context.Trace, traceNode, "edge_materialize_replayable", () => replayableSource.Materialize());
                        state = MeasurePhase(context.Trace, traceNode, "edge_build_replayable_buffer", () => BuildPathAwareEdgeStateFromRows(fallbackReplayableRows));
                    }
                    else
                    {
                        state = MeasurePhase(context.Trace, traceNode, "edge_build_external_materialized", () => BuildPathAwareEdgeStateFromRows(_provider.GetFacts(predicate) ?? Enumerable.Empty<object[]>()));
                    }
                    break;
            }

            context.PathAwareEdgeRetentionSelections[predicate] = selection;
            context.PathAwareEdgeStates[predicate] = state;
            return state;
        }

        private static PathAwareEdgeRetentionSelection GetCachedPathAwareEdgeRetentionSelection(PredicateId predicate, EvaluationContext context)
        {
            return context.PathAwareEdgeRetentionSelections.TryGetValue(predicate, out var selection)
                ? selection
                : new PathAwareEdgeRetentionSelection(PathAwareEdgeRetentionStrategy.Auto, "Unavailable");
        }

        private List<object[]> GetFactsList(PredicateId predicate, EvaluationContext? context)
        {
            if (context is null)
            {
                return MaterializeFacts(predicate, context);
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

            var facts = MaterializeFacts(predicate, context);
            context.Facts[predicate] = facts;
            return facts;
        }

        private HashSet<object[]> GetFactsSet(PredicateId predicate, EvaluationContext? context)
        {
            if (context is null)
            {
                return new HashSet<object[]>(MaterializeFacts(predicate, context), StructuralArrayComparer.Instance);
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
            var factSet = GetScanFactsSet(negation.Predicate, context, negation);

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
            var facts = GetScanFactsList(aggregate.Predicate, context, aggregate);
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

            var facts = GetScanFactsList(scan.Relation, context, scan);
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

                var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);
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

        private static RelationRetentionPolicyStrategy ToRelationRetentionPolicyStrategy(DagRelationRetentionStrategy strategy)
            => strategy switch
            {
                DagRelationRetentionStrategy.StreamingDirect => RelationRetentionPolicyStrategy.StreamingDirect,
                DagRelationRetentionStrategy.ReplayableBuffer => RelationRetentionPolicyStrategy.ReplayableBuffer,
                DagRelationRetentionStrategy.ExternalMaterialized => RelationRetentionPolicyStrategy.ExternalMaterialized,
                _ => RelationRetentionPolicyStrategy.Auto
            };

        private static DagRelationRetentionStrategy ToDagRelationRetentionStrategy(RelationRetentionPolicyStrategy strategy)
            => strategy switch
            {
                RelationRetentionPolicyStrategy.StreamingDirect => DagRelationRetentionStrategy.StreamingDirect,
                RelationRetentionPolicyStrategy.ReplayableBuffer => DagRelationRetentionStrategy.ReplayableBuffer,
                RelationRetentionPolicyStrategy.ExternalMaterialized => DagRelationRetentionStrategy.ExternalMaterialized,
                _ => DagRelationRetentionStrategy.Auto
            };

        private static RelationRetentionPolicyStrategy ToRelationRetentionPolicyStrategy(ScanRelationRetentionStrategy strategy)
            => strategy switch
            {
                ScanRelationRetentionStrategy.StreamingDirect => RelationRetentionPolicyStrategy.StreamingDirect,
                ScanRelationRetentionStrategy.ReplayableBuffer => RelationRetentionPolicyStrategy.ReplayableBuffer,
                ScanRelationRetentionStrategy.ExternalMaterialized => RelationRetentionPolicyStrategy.ExternalMaterialized,
                _ => RelationRetentionPolicyStrategy.Auto
            };

        private static ScanRelationRetentionStrategy ToScanRelationRetentionStrategy(RelationRetentionPolicyStrategy strategy)
            => strategy switch
            {
                RelationRetentionPolicyStrategy.StreamingDirect => ScanRelationRetentionStrategy.StreamingDirect,
                RelationRetentionPolicyStrategy.ReplayableBuffer => ScanRelationRetentionStrategy.ReplayableBuffer,
                RelationRetentionPolicyStrategy.ExternalMaterialized => ScanRelationRetentionStrategy.ExternalMaterialized,
                _ => ScanRelationRetentionStrategy.Auto
            };

        private static RelationRetentionPolicyStrategy ToRelationRetentionPolicyStrategy(ClosureRelationRetentionStrategy strategy)
            => strategy switch
            {
                ClosureRelationRetentionStrategy.StreamingDirect => RelationRetentionPolicyStrategy.StreamingDirect,
                ClosureRelationRetentionStrategy.ReplayableBuffer => RelationRetentionPolicyStrategy.ReplayableBuffer,
                ClosureRelationRetentionStrategy.ExternalMaterialized => RelationRetentionPolicyStrategy.ExternalMaterialized,
                _ => RelationRetentionPolicyStrategy.Auto
            };

        private static ClosureRelationRetentionStrategy ToClosureRelationRetentionStrategy(RelationRetentionPolicyStrategy strategy)
            => strategy switch
            {
                RelationRetentionPolicyStrategy.StreamingDirect => ClosureRelationRetentionStrategy.StreamingDirect,
                RelationRetentionPolicyStrategy.ReplayableBuffer => ClosureRelationRetentionStrategy.ReplayableBuffer,
                RelationRetentionPolicyStrategy.ExternalMaterialized => ClosureRelationRetentionStrategy.ExternalMaterialized,
                _ => ClosureRelationRetentionStrategy.Auto
            };

        private static RelationRetentionPolicyStrategy ToRelationRetentionPolicyStrategy(PathAwareEdgeRetentionStrategy strategy)
            => strategy switch
            {
                PathAwareEdgeRetentionStrategy.StreamingDirect => RelationRetentionPolicyStrategy.StreamingDirect,
                PathAwareEdgeRetentionStrategy.ReplayableBuffer => RelationRetentionPolicyStrategy.ReplayableBuffer,
                PathAwareEdgeRetentionStrategy.ExternalMaterialized => RelationRetentionPolicyStrategy.ExternalMaterialized,
                _ => RelationRetentionPolicyStrategy.Auto
            };

        private static PathAwareEdgeRetentionStrategy ToPathAwareEdgeRetentionStrategy(RelationRetentionPolicyStrategy strategy)
            => strategy switch
            {
                RelationRetentionPolicyStrategy.StreamingDirect => PathAwareEdgeRetentionStrategy.StreamingDirect,
                RelationRetentionPolicyStrategy.ReplayableBuffer => PathAwareEdgeRetentionStrategy.ReplayableBuffer,
                RelationRetentionPolicyStrategy.ExternalMaterialized => PathAwareEdgeRetentionStrategy.ExternalMaterialized,
                _ => PathAwareEdgeRetentionStrategy.Auto
            };

        private static RelationRetentionPolicyStrategy ToRelationRetentionPolicyStrategy(PathAwareSupportRelationRetentionStrategy strategy)
            => strategy switch
            {
                PathAwareSupportRelationRetentionStrategy.StreamingDirect => RelationRetentionPolicyStrategy.StreamingDirect,
                PathAwareSupportRelationRetentionStrategy.ReplayableBuffer => RelationRetentionPolicyStrategy.ReplayableBuffer,
                PathAwareSupportRelationRetentionStrategy.ExternalMaterialized => RelationRetentionPolicyStrategy.ExternalMaterialized,
                _ => RelationRetentionPolicyStrategy.Auto
            };

        private static PathAwareSupportRelationRetentionStrategy ToPathAwareSupportRelationRetentionStrategy(RelationRetentionPolicyStrategy strategy)
            => strategy switch
            {
                RelationRetentionPolicyStrategy.StreamingDirect => PathAwareSupportRelationRetentionStrategy.StreamingDirect,
                RelationRetentionPolicyStrategy.ReplayableBuffer => PathAwareSupportRelationRetentionStrategy.ReplayableBuffer,
                RelationRetentionPolicyStrategy.ExternalMaterialized => PathAwareSupportRelationRetentionStrategy.ExternalMaterialized,
                _ => PathAwareSupportRelationRetentionStrategy.Auto
            };

        private static RelationRetentionSelection ToRelationRetentionSelection(DagRelationRetentionSelection selection)
            => new(ToRelationRetentionPolicyStrategy(selection.Strategy), selection.DecisionMode);

        private static RelationRetentionSelection ToRelationRetentionSelection(ScanRelationRetentionSelection selection)
            => new(ToRelationRetentionPolicyStrategy(selection.Strategy), selection.DecisionMode);

        private static RelationRetentionSelection ToRelationRetentionSelection(ClosureRelationRetentionSelection selection)
            => new(ToRelationRetentionPolicyStrategy(selection.Strategy), selection.DecisionMode);

        private static RelationRetentionSelection ToRelationRetentionSelection(PathAwareEdgeRetentionSelection selection)
            => new(ToRelationRetentionPolicyStrategy(selection.Strategy), selection.DecisionMode);

        private static RelationRetentionSelection ToRelationRetentionSelection(PathAwareSupportRelationRetentionSelection selection)
            => new(ToRelationRetentionPolicyStrategy(selection.Strategy), selection.DecisionMode);

        private static RelationRetentionPolicyStrategy ResolveStructuralRelationRetentionStrategy(
            RelationRetentionPolicyStrategy preferredStrategy,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized)
        {
            if (replayableMaterialized && hasReplayable)
            {
                return RelationRetentionPolicyStrategy.ReplayableBuffer;
            }

            if (preferredStrategy == RelationRetentionPolicyStrategy.ReplayableBuffer)
            {
                if (hasReplayable)
                {
                    return RelationRetentionPolicyStrategy.ReplayableBuffer;
                }

                if (hasStreaming)
                {
                    return RelationRetentionPolicyStrategy.StreamingDirect;
                }
            }
            else
            {
                if (hasStreaming)
                {
                    return RelationRetentionPolicyStrategy.StreamingDirect;
                }

                if (hasReplayable)
                {
                    return RelationRetentionPolicyStrategy.ReplayableBuffer;
                }
            }

            return hasExternal
                ? RelationRetentionPolicyStrategy.ExternalMaterialized
                : preferredStrategy == RelationRetentionPolicyStrategy.ReplayableBuffer
                    ? RelationRetentionPolicyStrategy.ReplayableBuffer
                    : RelationRetentionPolicyStrategy.StreamingDirect;
        }

        private static bool ShouldProbeRelationRetentionStrategy(
            long? relationBytes,
            long thresholdBytes,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized)
        {
            if (replayableMaterialized)
            {
                return false;
            }

            var availableCount = (hasStreaming ? 1 : 0) + (hasReplayable ? 1 : 0) + (hasExternal ? 1 : 0);
            if (availableCount <= 1)
            {
                return false;
            }

            if (relationBytes is null)
            {
                return true;
            }

            return relationBytes <= thresholdBytes;
        }

        private static RelationRetentionPolicyStrategy ResolveMeasuredRelationRetentionStrategy(
            IReadOnlyDictionary<RelationRetentionPolicyStrategy, TimeSpan> probes,
            RelationRetentionPolicyStrategy structuralStrategy)
        {
            var bestStrategy = structuralStrategy;
            var bestTicks = double.PositiveInfinity;
            foreach (var entry in probes)
            {
                var ticks = entry.Value == TimeSpan.MaxValue
                    ? double.PositiveInfinity
                    : entry.Value.Ticks;
                if (ticks <= 0d || double.IsNaN(ticks))
                {
                    continue;
                }

                if (ticks < bestTicks)
                {
                    bestTicks = ticks;
                    bestStrategy = entry.Key;
                }
            }

            return double.IsInfinity(bestTicks) ? structuralStrategy : bestStrategy;
        }

        private static RelationRetentionSelection ResolveRelationRetentionStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            string selectPhase,
            string streamingProbePhase,
            string replayableProbePhase,
            string externalProbePhase,
            RelationRetentionPolicyStrategy configuredStrategy,
            RelationRetentionPolicyStrategy structuralStrategy,
            bool shouldProbe,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized,
            Func<TimeSpan>? measureStreamingProbe = null,
            Func<TimeSpan>? measureReplayableProbe = null,
            Func<TimeSpan>? measureExternalProbe = null)
        {
            return MeasurePhase(trace, node, selectPhase, () =>
            {
                bool IsAvailable(RelationRetentionPolicyStrategy strategy) => strategy switch
                {
                    RelationRetentionPolicyStrategy.StreamingDirect => hasStreaming,
                    RelationRetentionPolicyStrategy.ReplayableBuffer => hasReplayable,
                    RelationRetentionPolicyStrategy.ExternalMaterialized => hasExternal,
                    _ => hasStreaming || hasReplayable || hasExternal
                };

                if (configuredStrategy != RelationRetentionPolicyStrategy.Auto)
                {
                    if (IsAvailable(configuredStrategy))
                    {
                        return new RelationRetentionSelection(configuredStrategy, "ConfiguredOverride");
                    }

                    configuredStrategy = RelationRetentionPolicyStrategy.Auto;
                }

                if (replayableMaterialized && hasReplayable)
                {
                    return new RelationRetentionSelection(RelationRetentionPolicyStrategy.ReplayableBuffer, "ReplayableCached");
                }

                if (!shouldProbe)
                {
                    var availableCount = (hasStreaming ? 1 : 0) + (hasReplayable ? 1 : 0) + (hasExternal ? 1 : 0);
                    return new RelationRetentionSelection(structuralStrategy, availableCount <= 1 ? "OnlyAvailable" : "Structural");
                }

                var probes = new Dictionary<RelationRetentionPolicyStrategy, TimeSpan>();
                if (measureStreamingProbe is not null)
                {
                    var probe = measureStreamingProbe();
                    trace?.RecordPhase(node, streamingProbePhase, probe);
                    probes[RelationRetentionPolicyStrategy.StreamingDirect] = probe;
                }

                if (measureReplayableProbe is not null)
                {
                    var probe = measureReplayableProbe();
                    trace?.RecordPhase(node, replayableProbePhase, probe);
                    probes[RelationRetentionPolicyStrategy.ReplayableBuffer] = probe;
                }

                if (measureExternalProbe is not null)
                {
                    var probe = measureExternalProbe();
                    trace?.RecordPhase(node, externalProbePhase, probe);
                    probes[RelationRetentionPolicyStrategy.ExternalMaterialized] = probe;
                }

                if (probes.Count == 0)
                {
                    return new RelationRetentionSelection(structuralStrategy, "Structural");
                }

                return new RelationRetentionSelection(
                    ResolveMeasuredRelationRetentionStrategy(probes, structuralStrategy),
                    "MeasuredProbe");
            });
        }

        private static void RecordRelationRetentionStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            RelationRetentionSelection selection,
            string selectionPrefix,
            string strategyPrefix)
        {
            trace?.RecordStrategy(node, $"{selectionPrefix}{selection.DecisionMode}");
            trace?.RecordStrategy(node, $"{strategyPrefix}{selection.Strategy}");
        }

        private static DagRelationRetentionStrategy ResolveStructuralDagRelationRetentionStrategy(
            PlanNode node,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized)
        {
            var preferredStrategy = node is SeedGroupedTransitiveClosureCountNode or SeedGroupedDagLongestDepthNode
                ? RelationRetentionPolicyStrategy.StreamingDirect
                : RelationRetentionPolicyStrategy.ReplayableBuffer;
            return ToDagRelationRetentionStrategy(
                ResolveStructuralRelationRetentionStrategy(
                    preferredStrategy,
                    hasStreaming,
                    hasReplayable,
                    hasExternal,
                    replayableMaterialized));
        }

        private static bool ShouldProbeDagRelationRetentionStrategy(
            PlanNode node,
            long? relationBytes,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized)
        {
            var thresholdBytes = node switch
            {
                SeedGroupedTransitiveClosureCountNode => 384 * 1024L,
                SeedGroupedDagLongestDepthNode => 320 * 1024L,
                _ => 256 * 1024L,
            };
            return ShouldProbeRelationRetentionStrategy(
                relationBytes,
                thresholdBytes,
                hasStreaming,
                hasReplayable,
                hasExternal,
                replayableMaterialized);
        }

        private static DagRelationRetentionSelection ResolveDagRelationRetentionStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            DagRelationRetentionStrategy configuredStrategy,
            long? relationBytes,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized,
            Func<TimeSpan>? measureStreamingProbe = null,
            Func<TimeSpan>? measureReplayableProbe = null,
            Func<TimeSpan>? measureExternalProbe = null)
        {
            var structuralStrategy = ToRelationRetentionPolicyStrategy(
                ResolveStructuralDagRelationRetentionStrategy(
                    node,
                    hasStreaming,
                    hasReplayable,
                    hasExternal,
                    replayableMaterialized));
            var selection = ResolveRelationRetentionStrategy(
                trace,
                node,
                "dag_strategy_select",
                "dag_probe_streaming_direct",
                "dag_probe_replayable_buffer",
                "dag_probe_external_materialized",
                ToRelationRetentionPolicyStrategy(configuredStrategy),
                structuralStrategy,
                ShouldProbeDagRelationRetentionStrategy(node, relationBytes, hasStreaming, hasReplayable, hasExternal, replayableMaterialized),
                hasStreaming,
                hasReplayable,
                hasExternal,
                replayableMaterialized,
                measureStreamingProbe,
                measureReplayableProbe,
                measureExternalProbe);
            return new DagRelationRetentionSelection(ToDagRelationRetentionStrategy(selection.Strategy), selection.DecisionMode);
        }

        private static void RecordDagRelationRetentionStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            DagRelationRetentionSelection selection)
        {
            RecordRelationRetentionStrategy(
                trace,
                node,
                new RelationRetentionSelection(ToRelationRetentionPolicyStrategy(selection.Strategy), selection.DecisionMode),
                "DagRelationRetentionSelection",
                "DagRelationRetention");
        }

        private static ScanRelationRetentionStrategy ResolveStructuralScanRelationRetentionStrategy(
            PlanNode node,
            ScanRelationAccessKind accessKind,
            long? relationBytes,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized)
        {
            if (replayableMaterialized && hasReplayable)
            {
                return ScanRelationRetentionStrategy.ReplayableBuffer;
            }

            if (accessKind == ScanRelationAccessKind.Set && hasExternal)
            {
                return ScanRelationRetentionStrategy.ExternalMaterialized;
            }

            if (node is RelationScanNode &&
                accessKind == ScanRelationAccessKind.Stream &&
                hasExternal &&
                relationBytes is not null &&
                relationBytes.Value <= 1024L * 1024L)
            {
                return ScanRelationRetentionStrategy.ExternalMaterialized;
            }

            if (node is KeyJoinNode &&
                accessKind == ScanRelationAccessKind.List &&
                hasExternal &&
                relationBytes is not null &&
                relationBytes.Value <= 4 * 1024L * 1024L)
            {
                return ScanRelationRetentionStrategy.ExternalMaterialized;
            }

            if (node is PatternScanNode &&
                accessKind == ScanRelationAccessKind.List &&
                hasExternal &&
                relationBytes is not null &&
                relationBytes.Value <= 4 * 1024L * 1024L)
            {
                return ScanRelationRetentionStrategy.ExternalMaterialized;
            }

            if (node is NegationNode &&
                accessKind == ScanRelationAccessKind.List &&
                hasExternal &&
                relationBytes is not null &&
                relationBytes.Value <= 4 * 1024L * 1024L)
            {
                return ScanRelationRetentionStrategy.ExternalMaterialized;
            }

            if (node is AggregateNode &&
                accessKind == ScanRelationAccessKind.List &&
                hasExternal &&
                relationBytes is not null &&
                relationBytes.Value >= 32 * 1024L &&
                relationBytes.Value <= 4 * 1024L * 1024L)
            {
                return ScanRelationRetentionStrategy.ExternalMaterialized;
            }

            var preferredStrategy = accessKind == ScanRelationAccessKind.Stream
                ? RelationRetentionPolicyStrategy.StreamingDirect
                : RelationRetentionPolicyStrategy.ReplayableBuffer;
            return ToScanRelationRetentionStrategy(
                ResolveStructuralRelationRetentionStrategy(
                    preferredStrategy,
                    hasStreaming,
                    hasReplayable,
                    hasExternal,
                    replayableMaterialized));
        }

        private static bool ShouldProbeScanRelationRetentionStrategy(
            PlanNode node,
            ScanRelationAccessKind accessKind,
            long? relationBytes,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized)
        {
            if (accessKind == ScanRelationAccessKind.Set && hasExternal)
            {
                return false;
            }

            if (node is RelationScanNode &&
                accessKind == ScanRelationAccessKind.Stream &&
                hasExternal &&
                relationBytes is not null &&
                relationBytes.Value <= 1024L * 1024L)
            {
                return false;
            }

            if (node is KeyJoinNode &&
                accessKind == ScanRelationAccessKind.List &&
                hasExternal &&
                relationBytes is not null &&
                relationBytes.Value <= 4 * 1024L * 1024L)
            {
                return false;
            }

            if (node is PatternScanNode &&
                accessKind == ScanRelationAccessKind.List &&
                hasExternal &&
                relationBytes is not null &&
                relationBytes.Value <= 4 * 1024L * 1024L)
            {
                return false;
            }

            if (node is NegationNode &&
                accessKind == ScanRelationAccessKind.List &&
                hasExternal &&
                relationBytes is not null &&
                relationBytes.Value <= 4 * 1024L * 1024L)
            {
                return false;
            }

            if (node is AggregateNode &&
                accessKind == ScanRelationAccessKind.List &&
                hasExternal &&
                relationBytes is not null &&
                relationBytes.Value >= 32 * 1024L &&
                relationBytes.Value <= 4 * 1024L * 1024L)
            {
                return false;
            }

            var thresholdBytes = accessKind switch
            {
                ScanRelationAccessKind.Stream => node is RelationScanNode ? 256 * 1024L : 192 * 1024L,
                ScanRelationAccessKind.Set => 4 * 1024L * 1024L,
                ScanRelationAccessKind.List when node is KeyJoinNode => 4 * 1024L * 1024L,
                ScanRelationAccessKind.List when node is PatternScanNode => 4 * 1024L * 1024L,
                ScanRelationAccessKind.List when node is AggregateNode => 2 * 1024L * 1024L,
                _ => 512 * 1024L,
            };
            return ShouldProbeRelationRetentionStrategy(
                relationBytes,
                thresholdBytes,
                hasStreaming,
                hasReplayable,
                hasExternal,
                replayableMaterialized);
        }

        private static ScanRelationRetentionSelection ResolveScanRelationRetentionStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            ScanRelationRetentionStrategy configuredStrategy,
            ScanRelationAccessKind accessKind,
            long? relationBytes,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized,
            Func<TimeSpan>? measureStreamingProbe = null,
            Func<TimeSpan>? measureReplayableProbe = null,
            Func<TimeSpan>? measureExternalProbe = null)
        {
            var structuralStrategy = ToRelationRetentionPolicyStrategy(
                ResolveStructuralScanRelationRetentionStrategy(
                    node,
                    accessKind,
                    relationBytes,
                    hasStreaming,
                    hasReplayable,
                    hasExternal,
                    replayableMaterialized));
            var selection = ResolveRelationRetentionStrategy(
                trace,
                node,
                "scan_strategy_select",
                "scan_probe_streaming_direct",
                "scan_probe_replayable_buffer",
                "scan_probe_external_materialized",
                ToRelationRetentionPolicyStrategy(configuredStrategy),
                structuralStrategy,
                ShouldProbeScanRelationRetentionStrategy(node, accessKind, relationBytes, hasStreaming, hasReplayable, hasExternal, replayableMaterialized),
                hasStreaming,
                hasReplayable,
                hasExternal,
                replayableMaterialized,
                measureStreamingProbe,
                measureReplayableProbe,
                measureExternalProbe);
            return new ScanRelationRetentionSelection(ToScanRelationRetentionStrategy(selection.Strategy), selection.DecisionMode);
        }

        private static void RecordScanRelationRetentionStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            ScanRelationRetentionSelection selection)
        {
            RecordRelationRetentionStrategy(
                trace,
                node,
                new RelationRetentionSelection(ToRelationRetentionPolicyStrategy(selection.Strategy), selection.DecisionMode),
                "ScanRelationRetentionSelection",
                "ScanRelationRetention");
        }

        private static ClosureRelationRetentionStrategy ResolveStructuralClosureRelationRetentionStrategy(
            ClosureRelationAccessKind accessKind,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized)
        {
            var preferredStrategy = accessKind == ClosureRelationAccessKind.Edge
                ? RelationRetentionPolicyStrategy.ReplayableBuffer
                : RelationRetentionPolicyStrategy.ReplayableBuffer;
            return ToClosureRelationRetentionStrategy(
                ResolveStructuralRelationRetentionStrategy(
                    preferredStrategy,
                    hasStreaming,
                    hasReplayable,
                    hasExternal,
                    replayableMaterialized));
        }

        private static bool ShouldProbeClosureRelationRetentionStrategy(
            ClosureRelationAccessKind accessKind,
            long? relationBytes,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized)
        {
            var thresholdBytes = accessKind switch
            {
                ClosureRelationAccessKind.Edge => 384 * 1024L,
                _ => 256 * 1024L,
            };
            return ShouldProbeRelationRetentionStrategy(
                relationBytes,
                thresholdBytes,
                hasStreaming,
                hasReplayable,
                hasExternal,
                replayableMaterialized);
        }

        private static ClosureRelationRetentionSelection ResolveClosureRelationRetentionStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            ClosureRelationRetentionStrategy configuredStrategy,
            ClosureRelationAccessKind accessKind,
            long? relationBytes,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized,
            Func<TimeSpan>? measureStreamingProbe = null,
            Func<TimeSpan>? measureReplayableProbe = null,
            Func<TimeSpan>? measureExternalProbe = null)
        {
            var structuralStrategy = ToRelationRetentionPolicyStrategy(
                ResolveStructuralClosureRelationRetentionStrategy(
                    accessKind,
                    hasStreaming,
                    hasReplayable,
                    hasExternal,
                    replayableMaterialized));
            var selection = ResolveRelationRetentionStrategy(
                trace,
                node,
                "closure_strategy_select",
                "closure_probe_streaming_direct",
                "closure_probe_replayable_buffer",
                "closure_probe_external_materialized",
                ToRelationRetentionPolicyStrategy(configuredStrategy),
                structuralStrategy,
                ShouldProbeClosureRelationRetentionStrategy(accessKind, relationBytes, hasStreaming, hasReplayable, hasExternal, replayableMaterialized),
                hasStreaming,
                hasReplayable,
                hasExternal,
                replayableMaterialized,
                measureStreamingProbe,
                measureReplayableProbe,
                measureExternalProbe);
            return new ClosureRelationRetentionSelection(ToClosureRelationRetentionStrategy(selection.Strategy), selection.DecisionMode);
        }

        private static void RecordClosureRelationRetentionStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            ClosureRelationRetentionSelection selection)
        {
            RecordRelationRetentionStrategy(
                trace,
                node,
                new RelationRetentionSelection(ToRelationRetentionPolicyStrategy(selection.Strategy), selection.DecisionMode),
                "ClosureRelationRetentionSelection",
                "ClosureRelationRetention");
        }

        private static PathAwareSupportRelationRetentionStrategy ResolveStructuralPathAwareSupportRelationRetentionStrategy(
            PathAwareSupportRelationAccessKind accessKind,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized)
        {
            var preferredStrategy = accessKind switch
            {
                PathAwareSupportRelationAccessKind.Seeds => RelationRetentionPolicyStrategy.StreamingDirect,
                _ => RelationRetentionPolicyStrategy.StreamingDirect,
            };
            return ToPathAwareSupportRelationRetentionStrategy(
                ResolveStructuralRelationRetentionStrategy(
                    preferredStrategy,
                    hasStreaming,
                    hasReplayable,
                    hasExternal,
                    replayableMaterialized));
        }

        private static bool ShouldProbePathAwareSupportRelationRetentionStrategy(
            PathAwareSupportRelationAccessKind accessKind,
            long? relationBytes,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized)
        {
            var thresholdBytes = accessKind switch
            {
                PathAwareSupportRelationAccessKind.Roots => 128 * 1024L,
                _ => 256 * 1024L,
            };
            return ShouldProbeRelationRetentionStrategy(
                relationBytes,
                thresholdBytes,
                hasStreaming,
                hasReplayable,
                hasExternal,
                replayableMaterialized);
        }

        private static PathAwareSupportRelationRetentionSelection ResolvePathAwareSupportRelationRetentionStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            PathAwareSupportRelationRetentionStrategy configuredStrategy,
            PathAwareSupportRelationAccessKind accessKind,
            long? relationBytes,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized,
            Func<TimeSpan>? measureStreamingProbe = null,
            Func<TimeSpan>? measureReplayableProbe = null,
            Func<TimeSpan>? measureExternalProbe = null)
        {
            var structuralStrategy = ToRelationRetentionPolicyStrategy(
                ResolveStructuralPathAwareSupportRelationRetentionStrategy(
                    accessKind,
                    hasStreaming,
                    hasReplayable,
                    hasExternal,
                    replayableMaterialized));
            var selection = ResolveRelationRetentionStrategy(
                trace,
                node,
                accessKind == PathAwareSupportRelationAccessKind.Roots ? "support_roots_strategy_select" : "support_seeds_strategy_select",
                accessKind == PathAwareSupportRelationAccessKind.Roots ? "support_roots_probe_streaming_direct" : "support_seeds_probe_streaming_direct",
                accessKind == PathAwareSupportRelationAccessKind.Roots ? "support_roots_probe_replayable_buffer" : "support_seeds_probe_replayable_buffer",
                accessKind == PathAwareSupportRelationAccessKind.Roots ? "support_roots_probe_external_materialized" : "support_seeds_probe_external_materialized",
                ToRelationRetentionPolicyStrategy(configuredStrategy),
                structuralStrategy,
                ShouldProbePathAwareSupportRelationRetentionStrategy(accessKind, relationBytes, hasStreaming, hasReplayable, hasExternal, replayableMaterialized),
                hasStreaming,
                hasReplayable,
                hasExternal,
                replayableMaterialized,
                measureStreamingProbe,
                measureReplayableProbe,
                measureExternalProbe);
            return new PathAwareSupportRelationRetentionSelection(ToPathAwareSupportRelationRetentionStrategy(selection.Strategy), selection.DecisionMode);
        }

        private static void RecordPathAwareSupportRelationRetentionStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            PathAwareSupportRelationAccessKind accessKind,
            PathAwareSupportRelationRetentionSelection selection)
        {
            RecordRelationRetentionStrategy(
                trace,
                node,
                new RelationRetentionSelection(ToRelationRetentionPolicyStrategy(selection.Strategy), selection.DecisionMode),
                accessKind == PathAwareSupportRelationAccessKind.Roots
                    ? "PathAwareSupportRootsRelationRetentionSelection"
                    : "PathAwareSupportSeedsRelationRetentionSelection",
                accessKind == PathAwareSupportRelationAccessKind.Roots
                    ? "PathAwareSupportRootsRelationRetention"
                    : "PathAwareSupportSeedsRelationRetention");
        }

        private static PathAwareEdgeRetentionStrategy ResolveStructuralPathAwareEdgeRetentionStrategy(
            PlanNode node,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized)
        {
            var preferredStrategy = node is SeedGroupedPathAwareWeightSumNode
                ? RelationRetentionPolicyStrategy.ReplayableBuffer
                : RelationRetentionPolicyStrategy.StreamingDirect;
            return ToPathAwareEdgeRetentionStrategy(
                ResolveStructuralRelationRetentionStrategy(
                    preferredStrategy,
                    hasStreaming,
                    hasReplayable,
                    hasExternal,
                    replayableMaterialized));
        }

        private static bool ShouldProbePathAwareEdgeRetentionStrategy(
            PlanNode node,
            long? relationBytes,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized)
        {
            var thresholdBytes = node switch
            {
                SeedGroupedPathAwareWeightSumNode => 384 * 1024L,
                SeedGroupedPathAwareDepthMinNode or SeedGroupedPathAwareAccumulationMinNode => 320 * 1024L,
                _ => 256 * 1024L,
            };
            return ShouldProbeRelationRetentionStrategy(
                relationBytes,
                thresholdBytes,
                hasStreaming,
                hasReplayable,
                hasExternal,
                replayableMaterialized);
        }

        private static PathAwareEdgeRetentionSelection ResolvePathAwareEdgeRetentionStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            PathAwareEdgeRetentionStrategy configuredStrategy,
            long? relationBytes,
            bool hasStreaming,
            bool hasReplayable,
            bool hasExternal,
            bool replayableMaterialized,
            Func<TimeSpan>? measureStreamingProbe = null,
            Func<TimeSpan>? measureReplayableProbe = null,
            Func<TimeSpan>? measureExternalProbe = null)
        {
            var structuralStrategy = ToRelationRetentionPolicyStrategy(
                ResolveStructuralPathAwareEdgeRetentionStrategy(
                    node,
                    hasStreaming,
                    hasReplayable,
                    hasExternal,
                    replayableMaterialized));
            var selection = ResolveRelationRetentionStrategy(
                trace,
                node,
                "edge_strategy_select",
                "edge_probe_streaming_direct",
                "edge_probe_replayable_buffer",
                "edge_probe_external_materialized",
                ToRelationRetentionPolicyStrategy(configuredStrategy),
                structuralStrategy,
                ShouldProbePathAwareEdgeRetentionStrategy(node, relationBytes, hasStreaming, hasReplayable, hasExternal, replayableMaterialized),
                hasStreaming,
                hasReplayable,
                hasExternal,
                replayableMaterialized,
                measureStreamingProbe,
                measureReplayableProbe,
                measureExternalProbe);
            return new PathAwareEdgeRetentionSelection(ToPathAwareEdgeRetentionStrategy(selection.Strategy), selection.DecisionMode);
        }

        private static void RecordPathAwareEdgeRetentionStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            PathAwareEdgeRetentionSelection selection)
        {
            RecordRelationRetentionStrategy(
                trace,
                node,
                new RelationRetentionSelection(ToRelationRetentionPolicyStrategy(selection.Strategy), selection.DecisionMode),
                "PathAwareEdgeRetentionSelection",
                "PathAwareEdgeRetention");
        }

        private static PathAwareGroupedSummaryStrategy ToPathAwareGroupedSummaryStrategy(PathAwareGroupedMinStrategy strategy)
            => strategy switch
            {
                PathAwareGroupedMinStrategy.CompactGrouped => PathAwareGroupedSummaryStrategy.CompactGrouped,
                PathAwareGroupedMinStrategy.LegacySeededRows => PathAwareGroupedSummaryStrategy.LegacySeededRows,
                _ => PathAwareGroupedSummaryStrategy.Auto
            };

        private static PathAwareGroupedSummaryStrategy ToPathAwareGroupedSummaryStrategy(PathAwareWeightSumStrategy strategy)
            => strategy switch
            {
                PathAwareWeightSumStrategy.CompactGrouped => PathAwareGroupedSummaryStrategy.CompactGrouped,
                PathAwareWeightSumStrategy.LegacySeededRows => PathAwareGroupedSummaryStrategy.LegacySeededRows,
                _ => PathAwareGroupedSummaryStrategy.Auto
            };

        private static TimeSpan MeasureElapsed(Action action)
        {
            var stopwatch = Stopwatch.StartNew();
            action();
            stopwatch.Stop();
            return stopwatch.Elapsed;
        }

        private static T MeasurePhase<T>(QueryExecutionTrace? trace, PlanNode node, string phase, Func<T> action)
        {
            var stopwatch = Stopwatch.StartNew();
            var result = action();
            stopwatch.Stop();
            trace?.RecordPhase(node, phase, stopwatch.Elapsed);
            return result;
        }

        private static void MeasurePhase(QueryExecutionTrace? trace, PlanNode node, string phase, Action action)
        {
            var stopwatch = Stopwatch.StartNew();
            action();
            stopwatch.Stop();
            trace?.RecordPhase(node, phase, stopwatch.Elapsed);
        }

        private static long AdjustEstimatedLegacyRowsForPathAwareEdgeRetention(
            long estimatedGroupedRows,
            long estimatedLegacyRows,
            PathAwareEdgeRetentionStrategy edgeRetentionStrategy)
        {
            var factor = edgeRetentionStrategy switch
            {
                PathAwareEdgeRetentionStrategy.StreamingDirect => 1.50d,
                PathAwareEdgeRetentionStrategy.ReplayableBuffer => 0.75d,
                PathAwareEdgeRetentionStrategy.ExternalMaterialized => 0.90d,
                _ => 1.0d,
            };

            var adjusted = (long)Math.Ceiling(Math.Min(long.MaxValue, estimatedLegacyRows * factor));
            return Math.Max(estimatedGroupedRows, adjusted);
        }

        private static PathAwareGroupedSummaryStrategy ResolveStructuralPathAwareGroupedSummaryStrategy(
            long estimatedGroupedRows,
            long estimatedLegacyRows,
            PathAwareEdgeRetentionStrategy edgeRetentionStrategy)
        {
            var adjustedLegacyRows = AdjustEstimatedLegacyRowsForPathAwareEdgeRetention(
                estimatedGroupedRows,
                estimatedLegacyRows,
                edgeRetentionStrategy);
            return adjustedLegacyRows <= Math.Max(64L, estimatedGroupedRows * 4L)
                ? PathAwareGroupedSummaryStrategy.LegacySeededRows
                : PathAwareGroupedSummaryStrategy.CompactGrouped;
        }

        private static bool ShouldProbePathAwareGroupedSummaryStrategy(
            long estimatedGroupedRows,
            long estimatedLegacyRows,
            int uniqueSeedCount,
            PathAwareEdgeRetentionStrategy edgeRetentionStrategy)
        {
            if (uniqueSeedCount < 2)
            {
                return false;
            }

            var adjustedLegacyRows = AdjustEstimatedLegacyRowsForPathAwareEdgeRetention(
                estimatedGroupedRows,
                estimatedLegacyRows,
                edgeRetentionStrategy);
            var lowerBound = Math.Max(64L, estimatedGroupedRows * 2L);
            var upperBound = Math.Max(256L, estimatedGroupedRows * 8L);
            return adjustedLegacyRows > lowerBound && adjustedLegacyRows < upperBound;
        }

        private static PathAwareGroupedSummaryStrategy ResolveMeasuredPathAwareGroupedSummaryStrategy(
            int totalSeedCount,
            int sampleSeedCount,
            TimeSpan compactProbe,
            TimeSpan legacyProbe,
            long estimatedGroupedRows,
            long estimatedLegacyRows,
            PathAwareEdgeRetentionStrategy edgeRetentionStrategy)
        {
            var structuralStrategy = ResolveStructuralPathAwareGroupedSummaryStrategy(
                estimatedGroupedRows,
                estimatedLegacyRows,
                edgeRetentionStrategy);
            if (sampleSeedCount <= 0 || totalSeedCount <= 0)
            {
                return structuralStrategy;
            }

            var compactTicks = compactProbe == TimeSpan.MaxValue
                ? double.PositiveInfinity
                : compactProbe.Ticks * (double)totalSeedCount / sampleSeedCount;
            var legacyTicks = legacyProbe == TimeSpan.MaxValue
                ? double.PositiveInfinity
                : legacyProbe.Ticks * (double)totalSeedCount / sampleSeedCount;

            if (double.IsInfinity(compactTicks) && !double.IsInfinity(legacyTicks))
            {
                return PathAwareGroupedSummaryStrategy.LegacySeededRows;
            }

            if (double.IsInfinity(legacyTicks) && !double.IsInfinity(compactTicks))
            {
                return PathAwareGroupedSummaryStrategy.CompactGrouped;
            }

            if (compactTicks <= 0d || legacyTicks <= 0d || (double.IsInfinity(compactTicks) && double.IsInfinity(legacyTicks)))
            {
                return structuralStrategy;
            }

            return legacyTicks <= compactTicks
                ? PathAwareGroupedSummaryStrategy.LegacySeededRows
                : PathAwareGroupedSummaryStrategy.CompactGrouped;
        }

        private static PathAwareGroupedSummarySelection ResolvePathAwareGroupedSummaryStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            PathAwareGroupedSummaryStrategy configuredStrategy,
            int groupCount,
            IReadOnlyList<object?> orderedUniqueSeeds,
            int rootCount,
            IReadOnlyDictionary<object, PathAwareSuccessorBucket> succIndex,
            PathAwareEdgeRetentionStrategy edgeRetentionStrategy,
            Func<IReadOnlyList<object?>, TimeSpan>? measureCompactProbe = null,
            Func<IReadOnlyList<object?>, TimeSpan>? measureLegacyProbe = null)
        {
            return MeasurePhase(trace, node, "strategy_select", () =>
            {
                if (configuredStrategy != PathAwareGroupedSummaryStrategy.Auto)
                {
                    return new PathAwareGroupedSummarySelection(configuredStrategy, "ConfiguredOverride");
                }

                var uniqueSeedCount = orderedUniqueSeeds.Count;
                var effectiveGroups = Math.Max(1, groupCount);
                var effectiveSeeds = Math.Max(1, uniqueSeedCount);
                var effectiveRoots = Math.Max(1, rootCount);
                var nodeCount = Math.Max(1, EstimatePathAwareNodeCount(succIndex));
                var estimatedGroupedRows = (long)effectiveGroups * effectiveRoots;
                var estimatedLegacyRows = (long)effectiveSeeds * nodeCount;
                var structuralStrategy = ResolveStructuralPathAwareGroupedSummaryStrategy(
                    estimatedGroupedRows,
                    estimatedLegacyRows,
                    edgeRetentionStrategy);

                if (measureCompactProbe is null || measureLegacyProbe is null ||
                    !ShouldProbePathAwareGroupedSummaryStrategy(estimatedGroupedRows, estimatedLegacyRows, uniqueSeedCount, edgeRetentionStrategy))
                {
                    return new PathAwareGroupedSummarySelection(structuralStrategy, "Structural");
                }

                var sampleSeedCount = Math.Min(uniqueSeedCount, 4);
                var sampleSeeds = orderedUniqueSeeds.Take(sampleSeedCount).ToList();
                var compactProbe = measureCompactProbe(sampleSeeds);
                trace?.RecordPhase(node, "strategy_probe_compact_grouped", compactProbe);
                var legacyProbe = measureLegacyProbe(sampleSeeds);
                trace?.RecordPhase(node, "strategy_probe_legacy_seeded_rows", legacyProbe);

                var measuredStrategy = ResolveMeasuredPathAwareGroupedSummaryStrategy(
                    uniqueSeedCount,
                    sampleSeedCount,
                    compactProbe,
                    legacyProbe,
                    estimatedGroupedRows,
                    estimatedLegacyRows,
                    edgeRetentionStrategy);
                return new PathAwareGroupedSummarySelection(measuredStrategy, "MeasuredProbe");
            });
        }

        private static void RecordPathAwareGroupedSummaryStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            string nodeStrategyPrefix,
            PathAwareGroupedSummarySelection selection)
        {
            trace?.RecordStrategy(node, $"{nodeStrategyPrefix}Selection{selection.DecisionMode}");
            trace?.RecordStrategy(node, selection.Strategy == PathAwareGroupedSummaryStrategy.LegacySeededRows
                ? $"{nodeStrategyPrefix}LegacySeededRows"
                : $"{nodeStrategyPrefix}CompactGrouped");
        }

        private static string ResolveMaterializationPlanDecisionMode(
            RelationRetentionSelection relationRetentionSelection,
            PathAwareGroupedSummarySelection? groupedSummarySelection)
        {
            if (groupedSummarySelection is { } groupedSummary)
            {
                return groupedSummary.DecisionMode switch
                {
                    "ConfiguredOverride" => "ConfiguredSummary",
                    "MeasuredProbe" when relationRetentionSelection.DecisionMode == "MeasuredProbe" => "MeasuredRelationAndSummary",
                    "MeasuredProbe" => "MeasuredSummary",
                    _ when relationRetentionSelection.DecisionMode == "ConfiguredOverride" => "ConfiguredRelation",
                    _ when relationRetentionSelection.DecisionMode == "MeasuredProbe" => "MeasuredRelation",
                    _ when relationRetentionSelection.DecisionMode == "ReplayableCached" => "ReplayableCached",
                    _ when relationRetentionSelection.DecisionMode == "OnlyAvailable" => "OnlyAvailable",
                    _ => "Structural"
                };
            }

            return relationRetentionSelection.DecisionMode switch
            {
                "ConfiguredOverride" => "ConfiguredRelation",
                "MeasuredProbe" => "MeasuredRelation",
                "ReplayableCached" => "ReplayableCached",
                "OnlyAvailable" => "OnlyAvailable",
                _ => "Structural"
            };
        }

        private static MaterializationPlanSelection ResolveMaterializationPlan(
            QueryExecutionTrace? trace,
            PlanNode node,
            RelationRetentionSelection relationRetentionSelection,
            Func<PathAwareGroupedSummarySelection>? resolveGroupedSummarySelection = null)
        {
            return MeasurePhase(trace, node, "materialization_plan_select", () =>
            {
                var groupedSummarySelection = resolveGroupedSummarySelection?.Invoke();
                return new MaterializationPlanSelection(
                    relationRetentionSelection,
                    groupedSummarySelection,
                    ResolveMaterializationPlanDecisionMode(relationRetentionSelection, groupedSummarySelection));
            });
        }

        private static void RecordMaterializationPlanStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            string nodeStrategyPrefix,
            MaterializationPlanSelection selection)
        {
            trace?.RecordStrategy(node, $"{nodeStrategyPrefix}MaterializationPlanSelection{selection.DecisionMode}");
            if (selection.GroupedSummary is { } groupedSummary)
            {
                trace?.RecordStrategy(node,
                    $"{nodeStrategyPrefix}MaterializationPlanRelation{selection.RelationRetention.Strategy}Summary{groupedSummary.Strategy}");
            }
            else
            {
                trace?.RecordStrategy(node, $"{nodeStrategyPrefix}MaterializationPlanRelation{selection.RelationRetention.Strategy}");
            }
        }

        private static ClosurePairPlanStrategy? ResolveConfiguredClosurePairStrategy(
            ClosurePairStrategy configuredStrategy,
            bool singleConcretePairRequest,
            bool canBuildDirectionBatches)
        {
            return configuredStrategy switch
            {
                ClosurePairStrategy.Auto => null,
                ClosurePairStrategy.Forward => singleConcretePairRequest
                    ? ClosurePairPlanStrategy.SingleProbeForward
                    : ClosurePairPlanStrategy.Forward,
                ClosurePairStrategy.Backward => singleConcretePairRequest
                    ? ClosurePairPlanStrategy.SingleProbeBackward
                    : ClosurePairPlanStrategy.Backward,
                ClosurePairStrategy.MemoizedBySource => ClosurePairPlanStrategy.MemoizedBySource,
                ClosurePairStrategy.MemoizedByTarget => ClosurePairPlanStrategy.MemoizedByTarget,
                ClosurePairStrategy.MixedDirection when canBuildDirectionBatches => ClosurePairPlanStrategy.MixedDirection,
                ClosurePairStrategy.MixedDirectionWithPairProbeCache when canBuildDirectionBatches => ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache,
                _ => null
            };
        }

        private static ClosurePairPlanStrategy ResolveStructuralClosurePairStrategy(
            int sourceRequestCount,
            int targetRequestCount,
            bool singleConcretePairRequest,
            bool preferForwardSingleProbe,
            bool hasForwardBatch,
            bool hasBackwardBatch,
            bool canUseBatchedPairProbeCache,
            bool canMemoizeForwardBatch,
            bool canMemoizeBackwardBatch,
            bool canMemoizePairs,
            bool preferForwardFallback,
            bool preferSeededClosureCachesForPairBatches)
        {
            if (singleConcretePairRequest)
            {
                return preferForwardSingleProbe
                    ? ClosurePairPlanStrategy.SingleProbeForward
                    : ClosurePairPlanStrategy.SingleProbeBackward;
            }

            if (sourceRequestCount == 1 && targetRequestCount > 1 && canMemoizePairs)
            {
                return preferSeededClosureCachesForPairBatches
                    ? ClosurePairPlanStrategy.MemoizedBySource
                    : ClosurePairPlanStrategy.Forward;
            }

            if (targetRequestCount == 1 && sourceRequestCount > 1 && canMemoizePairs)
            {
                if (preferSeededClosureCachesForPairBatches)
                {
                    return ClosurePairPlanStrategy.MemoizedByTarget;
                }

                return hasBackwardBatch
                    ? ClosurePairPlanStrategy.Backward
                    : ClosurePairPlanStrategy.MemoizedByTarget;
            }

            if (hasForwardBatch && hasBackwardBatch)
            {
                return canUseBatchedPairProbeCache
                    ? ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache
                    : ClosurePairPlanStrategy.MixedDirection;
            }

            if (hasForwardBatch)
            {
                if (canUseBatchedPairProbeCache)
                {
                    return ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache;
                }

                return canMemoizeForwardBatch
                    ? ClosurePairPlanStrategy.MemoizedBySource
                    : ClosurePairPlanStrategy.Forward;
            }

            if (hasBackwardBatch)
            {
                if (canUseBatchedPairProbeCache)
                {
                    return ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache;
                }

                return canMemoizeBackwardBatch
                    ? ClosurePairPlanStrategy.MemoizedByTarget
                    : ClosurePairPlanStrategy.Backward;
            }

            if (canMemoizePairs)
            {
                return preferForwardFallback
                    ? ClosurePairPlanStrategy.MemoizedBySource
                    : ClosurePairPlanStrategy.MemoizedByTarget;
            }

            return preferForwardFallback
                ? ClosurePairPlanStrategy.Forward
                : ClosurePairPlanStrategy.Backward;
        }

        private static IReadOnlyList<ClosurePairPlanStrategy> DetermineClosurePairProbeStrategies(
            ClosurePairPlanStrategy structuralStrategy,
            int sourceRequestCount,
            int targetRequestCount,
            bool singleConcretePairRequest,
            bool hasForwardBatch,
            bool hasBackwardBatch,
            bool canUseBatchedPairProbeCache,
            bool canMemoizeForwardBatch,
            bool canMemoizeBackwardBatch,
            bool canMemoizePairs)
        {
            var strategies = new List<ClosurePairPlanStrategy>();

            void Add(ClosurePairPlanStrategy strategy)
            {
                if (!strategies.Contains(strategy))
                {
                    strategies.Add(strategy);
                }
            }

            Add(structuralStrategy);

            if (singleConcretePairRequest)
            {
                return strategies;
            }

            if (sourceRequestCount == 1 && targetRequestCount > 1)
            {
                Add(ClosurePairPlanStrategy.MemoizedBySource);
                Add(ClosurePairPlanStrategy.Forward);
                Add(ClosurePairPlanStrategy.Backward);
                if (hasForwardBatch && hasBackwardBatch)
                {
                    Add(ClosurePairPlanStrategy.MixedDirection);
                    if (canUseBatchedPairProbeCache)
                    {
                        Add(ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache);
                    }
                }

                return strategies;
            }

            if (targetRequestCount == 1 && sourceRequestCount > 1)
            {
                Add(ClosurePairPlanStrategy.MemoizedByTarget);
                Add(ClosurePairPlanStrategy.Backward);
                Add(ClosurePairPlanStrategy.Forward);
                if (hasForwardBatch && hasBackwardBatch)
                {
                    Add(ClosurePairPlanStrategy.MixedDirection);
                    if (canUseBatchedPairProbeCache)
                    {
                        Add(ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache);
                    }
                }

                return strategies;
            }

            if (hasForwardBatch && hasBackwardBatch &&
                (structuralStrategy == ClosurePairPlanStrategy.MixedDirection ||
                 structuralStrategy == ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache))
            {
                Add(ClosurePairPlanStrategy.MixedDirection);
                if (canUseBatchedPairProbeCache)
                {
                    Add(ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache);
                }

                return strategies;
            }

            if (hasForwardBatch && hasBackwardBatch)
            {
                Add(ClosurePairPlanStrategy.MixedDirection);
                if (canUseBatchedPairProbeCache)
                {
                    Add(ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache);
                }

                Add(ClosurePairPlanStrategy.Forward);
                Add(ClosurePairPlanStrategy.Backward);
                return strategies;
            }

            if (hasForwardBatch)
            {
                Add(ClosurePairPlanStrategy.Forward);
                if (canMemoizeForwardBatch || canMemoizePairs)
                {
                    Add(ClosurePairPlanStrategy.MemoizedBySource);
                }
                Add(ClosurePairPlanStrategy.Backward);
                return strategies;
            }

            if (hasBackwardBatch)
            {
                Add(ClosurePairPlanStrategy.Backward);
                if (canMemoizeBackwardBatch || canMemoizePairs)
                {
                    Add(ClosurePairPlanStrategy.MemoizedByTarget);
                }
                Add(ClosurePairPlanStrategy.Forward);
                return strategies;
            }

            if (canMemoizePairs)
            {
                Add(ClosurePairPlanStrategy.MemoizedBySource);
                Add(ClosurePairPlanStrategy.MemoizedByTarget);
            }

            Add(ClosurePairPlanStrategy.Forward);
            Add(ClosurePairPlanStrategy.Backward);
            return strategies;
        }

        private static bool ShouldProbeClosurePairStrategy(
            int requestCount,
            bool singleConcretePairRequest,
            IReadOnlyCollection<ClosurePairPlanStrategy> candidateStrategies)
        {
            if (singleConcretePairRequest || requestCount <= 1)
            {
                return false;
            }

            if (candidateStrategies.Count <= 1)
            {
                return false;
            }

            return requestCount <= 16;
        }

        private const double ClosurePairProbeOverrideMargin = 1.15d;

        private static ClosurePairPlanStrategy ResolveMeasuredClosurePairStrategy(
            IReadOnlyDictionary<ClosurePairPlanStrategy, TimeSpan> probes,
            ClosurePairPlanStrategy structuralStrategy)
        {
            if (probes.Count == 0)
            {
                return structuralStrategy;
            }

            var bestProbe = probes
                .Where(entry => entry.Value > TimeSpan.Zero && entry.Value < TimeSpan.MaxValue)
                .OrderBy(entry => entry.Value)
                .FirstOrDefault();

            if (bestProbe.Equals(default(KeyValuePair<ClosurePairPlanStrategy, TimeSpan>)) && !probes.ContainsKey(default))
            {
                return structuralStrategy;
            }

            if (!probes.TryGetValue(structuralStrategy, out var structuralProbe) ||
                structuralProbe <= TimeSpan.Zero ||
                structuralProbe == TimeSpan.MaxValue)
            {
                return bestProbe.Key;
            }

            if (bestProbe.Key == structuralStrategy)
            {
                return structuralStrategy;
            }

            return bestProbe.Value.Ticks * ClosurePairProbeOverrideMargin < structuralProbe.Ticks
                ? bestProbe.Key
                : structuralStrategy;
        }

        private static string GetClosurePairProbePhase(ClosurePairPlanStrategy strategy) => strategy switch
        {
            ClosurePairPlanStrategy.SingleProbeForward => "closure_pair_probe_single_forward",
            ClosurePairPlanStrategy.SingleProbeBackward => "closure_pair_probe_single_backward",
            ClosurePairPlanStrategy.Forward => "closure_pair_probe_forward",
            ClosurePairPlanStrategy.Backward => "closure_pair_probe_backward",
            ClosurePairPlanStrategy.MemoizedBySource => "closure_pair_probe_memo_source",
            ClosurePairPlanStrategy.MemoizedByTarget => "closure_pair_probe_memo_target",
            ClosurePairPlanStrategy.MixedDirection => "closure_pair_probe_mixed",
            ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache => "closure_pair_probe_mixed_cache",
            _ => "closure_pair_probe_unknown"
        };

        private static ClosurePairStrategySelection ResolveClosurePairStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            ClosurePairStrategy configuredStrategy,
            int requestCount,
            int sourceRequestCount,
            int targetRequestCount,
            bool singleConcretePairRequest,
            bool preferForwardSingleProbe,
            bool canBuildDirectionBatches,
            bool hasForwardBatch,
            bool hasBackwardBatch,
            bool canUseBatchedPairProbeCache,
            bool canMemoizeForwardBatch,
            bool canMemoizeBackwardBatch,
            bool canMemoizePairs,
            bool preferForwardFallback,
            bool preferSeededClosureCachesForPairBatches,
            IReadOnlyDictionary<ClosurePairPlanStrategy, Func<TimeSpan>>? measureProbes = null)
        {
            return MeasurePhase(trace, node, "closure_pair_strategy_select", () =>
            {
                var configured = ResolveConfiguredClosurePairStrategy(
                    configuredStrategy,
                    singleConcretePairRequest,
                    canBuildDirectionBatches);
                if (configured is { } configuredSelection)
                {
                    return new ClosurePairStrategySelection(configuredSelection, "ConfiguredOverride");
                }

                var structural = ResolveStructuralClosurePairStrategy(
                    sourceRequestCount,
                    targetRequestCount,
                    singleConcretePairRequest,
                    preferForwardSingleProbe,
                    hasForwardBatch,
                    hasBackwardBatch,
                    canUseBatchedPairProbeCache,
                    canMemoizeForwardBatch,
                    canMemoizeBackwardBatch,
                    canMemoizePairs,
                    preferForwardFallback,
                    preferSeededClosureCachesForPairBatches);

                if (configuredStrategy != ClosurePairStrategy.Auto)
                {
                    return new ClosurePairStrategySelection(structural, "ConfiguredFallback");
                }

                if (sourceRequestCount <= 1 || targetRequestCount <= 1)
                {
                    return new ClosurePairStrategySelection(structural, "Structural");
                }

                var candidateStrategies = DetermineClosurePairProbeStrategies(
                    structural,
                    sourceRequestCount,
                    targetRequestCount,
                    singleConcretePairRequest,
                    hasForwardBatch,
                    hasBackwardBatch,
                    canUseBatchedPairProbeCache,
                    canMemoizeForwardBatch,
                    canMemoizeBackwardBatch,
                    canMemoizePairs);

                if (!ShouldProbeClosurePairStrategy(requestCount, singleConcretePairRequest, candidateStrategies) ||
                    measureProbes is null ||
                    measureProbes.Count == 0)
                {
                    return new ClosurePairStrategySelection(structural, "Structural");
                }

                var probes = new Dictionary<ClosurePairPlanStrategy, TimeSpan>();
                foreach (var candidate in candidateStrategies)
                {
                    if (!measureProbes.TryGetValue(candidate, out var measureProbe))
                    {
                        continue;
                    }

                    var probe = measureProbe();
                    if (probe <= TimeSpan.Zero || probe == TimeSpan.MaxValue)
                    {
                        continue;
                    }

                    trace?.RecordPhase(node, GetClosurePairProbePhase(candidate), probe);
                    probes[candidate] = probe;
                }

                if (probes.Count == 0)
                {
                    return new ClosurePairStrategySelection(structural, "Structural");
                }

                return new ClosurePairStrategySelection(
                    ResolveMeasuredClosurePairStrategy(probes, structural),
                    "MeasuredProbe");
            });
        }

        private static void RecordClosurePairStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            string nodeStrategyPrefix,
            ClosurePairStrategySelection selection)
        {
            trace?.RecordStrategy(node, $"{nodeStrategyPrefix}MaterializationPlanSelection{selection.DecisionMode}");
            trace?.RecordStrategy(node, $"{nodeStrategyPrefix}MaterializationPlanPairs{selection.Strategy}");
        }

        private static int CountClosurePairRequests(IEnumerable<HashSet<object?>> requestSets) =>
            requestSets.Sum(requestSet => requestSet.Count);

        private const int MaxMeasuredClosurePairProbeRequests = 4;

        private static void TakeClosurePairRequestSample(
            IReadOnlyDictionary<object?, HashSet<object?>> bySource,
            int maxRequestCount,
            out Dictionary<object?, HashSet<object?>> sampleBySource,
            out Dictionary<object?, HashSet<object?>> sampleByTarget)
        {
            sampleBySource = new Dictionary<object?, HashSet<object?>>();
            sampleByTarget = new Dictionary<object?, HashSet<object?>>();
            var remaining = maxRequestCount;
            var active = bySource
                .Select(entry => (Source: entry.Key, Targets: entry.Value.GetEnumerator()))
                .ToList();

            while (remaining > 0 && active.Count > 0)
            {
                for (var i = 0; i < active.Count && remaining > 0;)
                {
                    var (source, targets) = active[i];
                    if (!targets.MoveNext())
                    {
                        active.RemoveAt(i);
                        continue;
                    }

                    var target = targets.Current;
                    if (!sampleBySource.TryGetValue(source, out var sampleTargets))
                    {
                        sampleTargets = new HashSet<object?>();
                        sampleBySource.Add(source, sampleTargets);
                    }

                    sampleTargets.Add(target);

                    if (!sampleByTarget.TryGetValue(target, out var sampleSources))
                    {
                        sampleSources = new HashSet<object?>();
                        sampleByTarget.Add(target, sampleSources);
                    }

                    sampleSources.Add(source);
                    remaining--;
                    i++;
                }
            }
        }

        private static void TakeGroupedClosurePairRequestSample(
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> targetsBySource,
            int maxRequestCount,
            out Dictionary<RowWrapper, HashSet<object?>> sampleTargetsBySource,
            out Dictionary<RowWrapper, HashSet<object?>> sampleSourcesByTarget)
        {
            var wrapperComparer = new RowWrapperComparer(StructuralArrayComparer.Instance);
            sampleTargetsBySource = new Dictionary<RowWrapper, HashSet<object?>>(wrapperComparer);
            sampleSourcesByTarget = new Dictionary<RowWrapper, HashSet<object?>>(wrapperComparer);
            var remaining = maxRequestCount;
            var active = targetsBySource
                .Select(entry => (SourceWrapper: entry.Key, Targets: entry.Value.GetEnumerator()))
                .ToList();

            while (remaining > 0 && active.Count > 0)
            {
                for (var i = 0; i < active.Count && remaining > 0;)
                {
                    var (sourceWrapper, targets) = active[i];
                    if (!targets.MoveNext())
                    {
                        active.RemoveAt(i);
                        continue;
                    }

                    var sourceRow = sourceWrapper.Row;
                    var source = sourceRow[sourceRow.Length - 1];
                    var target = targets.Current;
                    if (!sampleTargetsBySource.TryGetValue(sourceWrapper, out var sampleTargets))
                    {
                        sampleTargets = new HashSet<object?>();
                        sampleTargetsBySource.Add(sourceWrapper, sampleTargets);
                    }

                    sampleTargets.Add(target);

                    var targetRow = new object[sourceRow.Length];
                    Array.Copy(sourceRow, targetRow, sourceRow.Length - 1);
                    targetRow[sourceRow.Length - 1] = target;
                    var targetWrapper = new RowWrapper(targetRow);

                    if (!sampleSourcesByTarget.TryGetValue(targetWrapper, out var sampleSources))
                    {
                        sampleSources = new HashSet<object?>();
                        sampleSourcesByTarget.Add(targetWrapper, sampleSources);
                    }

                    sampleSources.Add(source);
                    remaining--;
                    i++;
                }
            }
        }

        private EvaluationContext CreateClosurePairProbeContext(EvaluationContext context)
        {
            var probeContext = new EvaluationContext(trace: new QueryExecutionTrace(), cancellationToken: context.CancellationToken)
            {
                Current = context.Current,
                FixpointDepth = context.FixpointDepth
            };

            foreach (var kvp in context.Facts)
            {
                probeContext.Facts[kvp.Key] = kvp.Value;
            }

            foreach (var kvp in context.FactSources)
            {
                probeContext.FactSources[kvp.Key] = kvp.Value;
            }

            foreach (var kvp in context.ReplayableFactSources)
            {
                probeContext.ReplayableFactSources[kvp.Key] = kvp.Value;
            }

            foreach (var kvp in context.ClosureRelationRetentionSelections)
            {
                probeContext.ClosureRelationRetentionSelections[kvp.Key] = kvp.Value;
            }

            foreach (var kvp in context.FactIndices)
            {
                probeContext.FactIndices[kvp.Key] = kvp.Value;
            }

            foreach (var kvp in context.JoinIndices)
            {
                probeContext.JoinIndices[kvp.Key] = kvp.Value;
            }

            return probeContext;
        }

        private TimeSpan MeasureGroupedClosurePairStrategyProbe(
            GroupedTransitiveClosureNode closure,
            ClosurePairPlanStrategy strategy,
            IReadOnlyList<int> inputPositions,
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> targetsBySource,
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> sourcesByTarget,
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> forwardTargetsBySource,
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> backwardSourcesByTarget,
            EvaluationContext context)
        {
            var sampleTargetsBySource = targetsBySource;
            var sampleSourcesByTarget = sourcesByTarget;
            var sampleForwardTargetsBySource = forwardTargetsBySource;
            var sampleBackwardSourcesByTarget = backwardSourcesByTarget;

            if (CountClosurePairRequests(targetsBySource.Values) > MaxMeasuredClosurePairProbeRequests)
            {
                TakeGroupedClosurePairRequestSample(
                    targetsBySource,
                    MaxMeasuredClosurePairProbeRequests,
                    out var limitedTargetsBySource,
                    out var limitedSourcesByTarget);
                sampleTargetsBySource = limitedTargetsBySource;
                sampleSourcesByTarget = limitedSourcesByTarget;

                var batchProbeContext = CreateClosurePairProbeContext(context);
                if (!TryBuildGroupedPairProbeDirectionBatches(
                    closure,
                    sampleTargetsBySource,
                    batchProbeContext,
                    out var limitedForwardTargetsBySource,
                    out var limitedBackwardSourcesByTarget))
                {
                    var wrapperComparer = new RowWrapperComparer(StructuralArrayComparer.Instance);
                    limitedForwardTargetsBySource = new Dictionary<RowWrapper, HashSet<object?>>(wrapperComparer);
                    limitedBackwardSourcesByTarget = new Dictionary<RowWrapper, HashSet<object?>>(wrapperComparer);
                }

                sampleForwardTargetsBySource = limitedForwardTargetsBySource;
                sampleBackwardSourcesByTarget = limitedBackwardSourcesByTarget;
            }

            return strategy switch
            {
                ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache when sampleForwardTargetsBySource.Count > 0 && sampleBackwardSourcesByTarget.Count > 0 => MeasureElapsed(() =>
                {
                    var probeContext = CreateClosurePairProbeContext(context);
                    _ = ExecuteSeededGroupedTransitiveClosurePairsMixedDirectionWithPairProbeCache(
                        closure,
                        inputPositions,
                        sampleForwardTargetsBySource,
                        sampleBackwardSourcesByTarget,
                        probeContext).ToList();
                }),
                ClosurePairPlanStrategy.MixedDirection when sampleForwardTargetsBySource.Count > 0 && sampleBackwardSourcesByTarget.Count > 0 => MeasureElapsed(() =>
                {
                    var probeContext = CreateClosurePairProbeContext(context);
                    _ = ExecuteSeededGroupedTransitiveClosurePairsMixedDirection(
                        closure,
                        inputPositions,
                        sampleForwardTargetsBySource,
                        sampleBackwardSourcesByTarget,
                        probeContext).ToList();
                }),
                ClosurePairPlanStrategy.MemoizedBySource => MeasureElapsed(() =>
                {
                    var probeContext = CreateClosurePairProbeContext(context);
                    _ = ExecuteSeededGroupedTransitiveClosurePairsMemoizedBySource(
                        closure,
                        inputPositions,
                        sampleTargetsBySource,
                        probeContext).ToList();
                }),
                ClosurePairPlanStrategy.MemoizedByTarget => MeasureElapsed(() =>
                {
                    var probeContext = CreateClosurePairProbeContext(context);
                    _ = ExecuteSeededGroupedTransitiveClosurePairsMemoizedByTarget(
                        closure,
                        inputPositions,
                        sampleSourcesByTarget,
                        probeContext).ToList();
                }),
                ClosurePairPlanStrategy.Forward => MeasureElapsed(() =>
                {
                    var probeContext = CreateClosurePairProbeContext(context);
                    _ = ExecuteSeededGroupedTransitiveClosurePairsForward(
                        closure,
                        sampleTargetsBySource,
                        probeContext).ToList();
                }),
                ClosurePairPlanStrategy.Backward => MeasureElapsed(() =>
                {
                    var probeContext = CreateClosurePairProbeContext(context);
                    _ = ExecuteSeededGroupedTransitiveClosurePairsBackward(
                        closure,
                        sampleSourcesByTarget,
                        probeContext).ToList();
                }),
                _ => TimeSpan.MaxValue
            };
        }

        private TimeSpan MeasureUngroupedClosurePairStrategyProbe(
            TransitiveClosureNode closure,
            ClosurePairPlanStrategy strategy,
            IReadOnlyDictionary<object?, HashSet<object?>> bySource,
            IReadOnlyDictionary<object?, HashSet<object?>> byTarget,
            IReadOnlyDictionary<object?, HashSet<object?>> forwardBySource,
            IReadOnlyDictionary<object?, HashSet<object?>> backwardByTarget,
            EvaluationContext context)
        {
            var sampleBySource = bySource;
            var sampleByTarget = byTarget;
            var sampleForwardBySource = forwardBySource;
            var sampleBackwardByTarget = backwardByTarget;

            if (CountClosurePairRequests(bySource.Values) > MaxMeasuredClosurePairProbeRequests)
            {
                TakeClosurePairRequestSample(
                    bySource,
                    MaxMeasuredClosurePairProbeRequests,
                    out var limitedBySource,
                    out var limitedByTarget);
                sampleBySource = limitedBySource;
                sampleByTarget = limitedByTarget;

                var batchProbeContext = CreateClosurePairProbeContext(context);
                if (!TryBuildPairProbeDirectionBatches(
                    closure,
                    sampleBySource,
                    batchProbeContext,
                    out var limitedForwardBySource,
                    out var limitedBackwardByTarget))
                {
                    limitedForwardBySource = new Dictionary<object?, HashSet<object?>>();
                    limitedBackwardByTarget = new Dictionary<object?, HashSet<object?>>();
                }

                sampleForwardBySource = limitedForwardBySource;
                sampleBackwardByTarget = limitedBackwardByTarget;
            }

            return strategy switch
            {
                ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache when sampleForwardBySource.Count > 0 && sampleBackwardByTarget.Count > 0 => MeasureElapsed(() =>
                {
                    var probeContext = CreateClosurePairProbeContext(context);
                    _ = ExecuteSeededTransitiveClosurePairsMixedDirectionWithPairProbeCache(
                        closure,
                        sampleForwardBySource,
                        sampleBackwardByTarget,
                        probeContext).ToList();
                }),
                ClosurePairPlanStrategy.MixedDirection when sampleForwardBySource.Count > 0 && sampleBackwardByTarget.Count > 0 => MeasureElapsed(() =>
                {
                    var probeContext = CreateClosurePairProbeContext(context);
                    _ = ExecuteSeededTransitiveClosurePairsMixedDirection(
                        closure,
                        sampleForwardBySource,
                        sampleBackwardByTarget,
                        probeContext).ToList();
                }),
                ClosurePairPlanStrategy.MemoizedBySource => MeasureElapsed(() =>
                {
                    var probeContext = CreateClosurePairProbeContext(context);
                    _ = ExecuteSeededTransitiveClosurePairsMemoizedBySource(
                        closure,
                        sampleBySource,
                        probeContext).ToList();
                }),
                ClosurePairPlanStrategy.MemoizedByTarget => MeasureElapsed(() =>
                {
                    var probeContext = CreateClosurePairProbeContext(context);
                    _ = ExecuteSeededTransitiveClosurePairsMemoizedByTarget(
                        closure,
                        sampleByTarget,
                        probeContext).ToList();
                }),
                ClosurePairPlanStrategy.Forward => MeasureElapsed(() =>
                {
                    var probeContext = CreateClosurePairProbeContext(context);
                    _ = ExecuteSeededTransitiveClosurePairsForward(
                        closure,
                        sampleBySource,
                        probeContext).ToList();
                }),
                ClosurePairPlanStrategy.Backward => MeasureElapsed(() =>
                {
                    var probeContext = CreateClosurePairProbeContext(context);
                    _ = ExecuteSeededTransitiveClosurePairsBackward(
                        closure,
                        sampleByTarget,
                        probeContext).ToList();
                }),
                _ => TimeSpan.MaxValue
            };
        }

        private IReadOnlyDictionary<object, Dictionary<object, double>> ExecuteSeedGroupedPathAwareWeightSumFallback(
            SeedGroupedPathAwareWeightSumNode closure,
            IReadOnlyList<object?> orderedUniqueSeeds,
            ISet<object> rootKeys,
            EvaluationContext context)
        {
            if (orderedUniqueSeeds.Count == 0)
            {
                return new Dictionary<object, Dictionary<object, double>>();
            }

            var seededClosure = new PathAwareTransitiveClosureNode(
                closure.EdgeRelation,
                closure.Predicate,
                1,
                1,
                closure.MaxDepth,
                TableMode.All);
            var seedParams = orderedUniqueSeeds.Select(seed => new object[] { seed! }).ToList();
            var rows = ExecuteSeededPathAwareTransitiveClosure(seededClosure, new[] { 0 }, seedParams, context).ToList();
            var distanceWeightCache = new Dictionary<int, double>();

            double GetDistanceWeight(int distance)
            {
                if (distanceWeightCache.TryGetValue(distance, out var cachedWeight))
                {
                    return cachedWeight;
                }

                var weight = Math.Pow(distance, -closure.DistanceExponent);
                distanceWeightCache[distance] = weight;
                return weight;
            }

            var seedWeightSums = new Dictionary<object, Dictionary<object, double>>();
            foreach (var seed in orderedUniqueSeeds)
            {
                var seedKey = seed ?? NullFactIndexKey;
                if (rootKeys.Contains(seedKey))
                {
                    seedWeightSums[seedKey] = new Dictionary<object, double>
                    {
                        [seedKey] = GetDistanceWeight(1)
                    };
                }
            }

            foreach (var row in rows)
            {
                if (row is null || row.Length < 3)
                {
                    continue;
                }

                var seed = row[0];
                var root = row[1];
                var rootKey = root ?? NullFactIndexKey;
                if (!rootKeys.Contains(rootKey))
                {
                    continue;
                }

                var seedKey = seed ?? NullFactIndexKey;
                if (!seedWeightSums.TryGetValue(seedKey, out var rootSums))
                {
                    rootSums = new Dictionary<object, double>();
                    seedWeightSums[seedKey] = rootSums;
                }

                var distance = Convert.ToInt32(row[2], CultureInfo.InvariantCulture);
                var weight = GetDistanceWeight(distance);
                rootSums[rootKey] = rootSums.TryGetValue(rootKey, out var current) ? current + weight : weight;
            }

            return seedWeightSums;
        }

        private IEnumerable<object[]> ExecuteSeedGroupedPathAwareWeightSum(SeedGroupedPathAwareWeightSumNode closure, EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));

            var width = closure.Predicate.Arity;
            if (width < 3)
            {
                return Array.Empty<object[]>();
            }

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                trace?.RecordStrategy(closure, "SeedGroupedPathAwareWeightSum");

                var predicate = closure.Predicate;
                var cacheKey = (closure.EdgeRelation, closure.SeedRelation, closure.RootRelation, predicate, closure.DistanceExponent, closure.MaxDepth);
                var traceKey = $"{predicate.Name}/{predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:seeds={closure.SeedRelation.Name}/{closure.SeedRelation.Arity}:roots={closure.RootRelation.Name}/{closure.RootRelation.Arity}:exp={closure.DistanceExponent.ToString(CultureInfo.InvariantCulture)}:max={closure.MaxDepth}";
                if (context.SeedGroupedPathAwareWeightSumResults.TryGetValue(cacheKey, out var cachedRows))
                {
                    trace?.RecordCacheLookup("SeedGroupedPathAwareWeightSum", traceKey, hit: true, built: false);
                    return cachedRows;
                }

                trace?.RecordCacheLookup("SeedGroupedPathAwareWeightSum", traceKey, hit: false, built: true);

                var edgeState = GetPathAwareEdgeState(closure.EdgeRelation, context, closure);
                var edgeRetentionSelection = GetCachedPathAwareEdgeRetentionSelection(closure.EdgeRelation, context);
                var succIndex = edgeState.Successors;

                var rootKeys = new HashSet<object>();
                var rootValues = new Dictionary<object, object?>();
                MeasurePhase(trace, closure, "load_roots", () =>
                {
                    foreach (var row in GetPathAwareSupportFactStream(closure.RootRelation, PathAwareSupportRelationAccessKind.Roots, context, closure))
                    {
                        if (row is null || row.Length == 0)
                        {
                            continue;
                        }

                        var root = row[0];
                        var rootKey = root ?? NullFactIndexKey;
                        if (rootKeys.Add(rootKey))
                        {
                            rootValues[rootKey] = root;
                        }
                    }
                });

                if (rootKeys.Count == 0)
                {
                    var empty = Array.Empty<object[]>();
                    context.SeedGroupedPathAwareWeightSumResults[cacheKey] = empty;
                    return empty;
                }

                var groupedSeeds = new Dictionary<object, List<object?>>();
                var groupValues = new List<object?>();
                var uniqueSeeds = new HashSet<object?>();
                var orderedUniqueSeeds = new List<object?>();
                MeasurePhase(trace, closure, "load_seeds", () =>
                {
                    foreach (var row in GetPathAwareSupportFactStream(closure.SeedRelation, PathAwareSupportRelationAccessKind.Seeds, context, closure))
                    {
                        if (row is null || row.Length < 2)
                        {
                            continue;
                        }

                        var group = row[0];
                        var seed = row[1];
                        var groupKey = group ?? NullFactIndexKey;
                        if (!groupedSeeds.TryGetValue(groupKey, out var bucket))
                        {
                            bucket = new List<object?>();
                            groupedSeeds[groupKey] = bucket;
                            groupValues.Add(group);
                        }

                        bucket.Add(seed);
                        if (uniqueSeeds.Add(seed))
                        {
                            orderedUniqueSeeds.Add(seed);
                        }
                    }
                });

                if (groupValues.Count == 0)
                {
                    var empty = Array.Empty<object[]>();
                    context.SeedGroupedPathAwareWeightSumResults[cacheKey] = empty;
                    return empty;
                }

                orderedUniqueSeeds.Sort(CompareCacheSeedValues);
                groupValues.Sort(CompareCacheSeedValues);

                Dictionary<object, Dictionary<object, double>> BuildCompactSeedWeightSums(IReadOnlyList<object?> seeds)
                {
                    var distanceWeightCache = new Dictionary<int, double>();
                    var nodeIds = new Dictionary<object, int>();
                    var nextNodeId = 0;

                    int GetNodeId(object? value)
                    {
                        var key = value ?? NullFactIndexKey;
                        if (nodeIds.TryGetValue(key, out var existing))
                        {
                            return existing;
                        }

                        var id = nextNodeId++;
                        nodeIds[key] = id;
                        return id;
                    }

                    double GetDistanceWeight(int distance)
                    {
                        if (distanceWeightCache.TryGetValue(distance, out var cachedWeight))
                        {
                            return cachedWeight;
                        }

                        var weight = Math.Pow(distance, -closure.DistanceExponent);
                        distanceWeightCache[distance] = weight;
                        return weight;
                    }

                    var directSeedWeightSums = new Dictionary<object, Dictionary<object, double>>();
                    foreach (var seed in seeds)
                    {
                        var weightSums = ComputePathAwareRootWeightSumsForSeed(seed, succIndex, rootKeys, closure.MaxDepth, GetNodeId, GetDistanceWeight);
                        if (weightSums.Count > 0)
                        {
                            directSeedWeightSums[seed ?? NullFactIndexKey] = weightSums;
                        }
                    }

                    return directSeedWeightSums;
                }

                TimeSpan MeasureCompactProbe(IReadOnlyList<object?> sampleSeeds) =>
                    MeasureElapsed(() => _ = BuildCompactSeedWeightSums(sampleSeeds));

                TimeSpan MeasureLegacyProbe(IReadOnlyList<object?> sampleSeeds) =>
                    MeasureElapsed(() => _ = ExecuteSeedGroupedPathAwareWeightSumFallback(closure, sampleSeeds, rootKeys, context));

                var materializationPlan = ResolveMaterializationPlan(
                    trace,
                    closure,
                    ToRelationRetentionSelection(edgeRetentionSelection),
                    () => ResolvePathAwareGroupedSummaryStrategy(
                        trace,
                        closure,
                        _pathAwareWeightSumStrategy,
                        groupValues.Count,
                        orderedUniqueSeeds,
                        rootKeys.Count,
                        succIndex,
                        edgeRetentionSelection.Strategy,
                        MeasureCompactProbe,
                        MeasureLegacyProbe));
                RecordMaterializationPlanStrategy(trace, closure, "SeedGroupedPathAwareWeightSum", materializationPlan);
                var groupedSummarySelection = materializationPlan.GroupedSummary
                    ?? throw new InvalidOperationException("Expected grouped-summary selection for SeedGroupedPathAwareWeightSumNode.");
                RecordPathAwareGroupedSummaryStrategy(trace, closure, "SeedGroupedPathAwareWeightSum", groupedSummarySelection);

                IReadOnlyDictionary<object, Dictionary<object, double>> seedWeightSums = groupedSummarySelection.Strategy == PathAwareGroupedSummaryStrategy.LegacySeededRows
                    ? MeasurePhase(trace, closure, "build_legacy_seeded_rows", () =>
                        (IReadOnlyDictionary<object, Dictionary<object, double>>)ExecuteSeedGroupedPathAwareWeightSumFallback(closure, orderedUniqueSeeds, rootKeys, context))
                    : MeasurePhase(trace, closure, "build_compact_grouped", () =>
                        (IReadOnlyDictionary<object, Dictionary<object, double>>)BuildCompactSeedWeightSums(orderedUniqueSeeds));

                var rows = MeasurePhase(trace, closure, "group_reduce", () =>
                {
                    var builtRows = new List<object[]>();
                    foreach (var group in groupValues)
                    {
                        var groupKey = group ?? NullFactIndexKey;
                        if (!groupedSeeds.TryGetValue(groupKey, out var seedsForGroup))
                        {
                            continue;
                        }

                        Dictionary<object, double>? groupSums = null;
                        foreach (var seed in seedsForGroup)
                        {
                            if (!seedWeightSums.TryGetValue(seed ?? NullFactIndexKey, out var rootSums))
                            {
                                continue;
                            }

                            groupSums ??= new Dictionary<object, double>();
                            foreach (var entry in rootSums)
                            {
                                groupSums[entry.Key] = groupSums.TryGetValue(entry.Key, out var current) ? current + entry.Value : entry.Value;
                            }
                        }

                        if (groupSums is null)
                        {
                            continue;
                        }

                        var orderedRoots = groupSums.Keys.ToList();
                        orderedRoots.Sort((left, right) => CompareCacheSeedValues(rootValues[left], rootValues[right]));
                        foreach (var rootKey in orderedRoots)
                        {
                            builtRows.Add(new object[] { group!, rootValues[rootKey]!, groupSums[rootKey] });
                        }
                    }

                    return builtRows;
                });

                context.SeedGroupedPathAwareWeightSumResults[cacheKey] = rows;
                return rows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private static Dictionary<object, double> ComputePathAwareRootWeightSumsForSeed(
            object? seed,
            IReadOnlyDictionary<object, PathAwareSuccessorBucket> succIndex,
            HashSet<object> rootKeys,
            int maxDepth,
            Func<object?, int> getNodeId,
            Func<int, double> getDistanceWeight)
        {
            var sums = new Dictionary<object, double>();

            void AddContribution(object? root, int distance)
            {
                var rootKey = root ?? NullFactIndexKey;
                var weight = getDistanceWeight(distance);
                sums[rootKey] = sums.TryGetValue(rootKey, out var current) ? current + weight : weight;
            }

            var seedKey = seed ?? NullFactIndexKey;
            if (rootKeys.Contains(seedKey))
            {
                AddContribution(seed, 1);
            }

            var initialPath = CompactVisitedPath.Create(getNodeId(seed));
            var stack = new Stack<(object? Node, int Depth, CompactVisitedPath Path)>();
            stack.Push((seed, 0, initialPath));

            while (stack.Count > 0)
            {
                var (current, depth, path) = stack.Pop();
                var lookupKey = current ?? NullFactIndexKey;
                if (!succIndex.TryGetValue(lookupKey, out var bucket))
                {
                    continue;
                }

                for (var i = bucket.Targets.Count - 1; i >= 0; i--)
                {
                    var next = bucket.Targets[i];
                    var nextId = getNodeId(next);
                    if (path.Contains(nextId))
                    {
                        continue;
                    }

                    var nextDepth = depth + 1;
                    if (maxDepth > 0 && nextDepth > maxDepth)
                    {
                        continue;
                    }

                    var nextKey = next ?? NullFactIndexKey;
                    if (rootKeys.Contains(nextKey))
                    {
                        AddContribution(next, nextDepth + 1);
                    }

                    stack.Push((next, nextDepth, path.Extend(nextId)));
                }
            }

            return sums;
        }



        private static int EstimatePathAwareNodeCount(IReadOnlyDictionary<object, PathAwareSuccessorBucket> succIndex)
        {
            var nodes = new HashSet<object>();
            foreach (var (key, bucket) in succIndex)
            {
                nodes.Add(key);
                foreach (var target in bucket.Targets)
                {
                    nodes.Add(target ?? NullFactIndexKey);
                }
            }

            return nodes.Count;
        }

        private static PathAwareSccGraph BuildPathAwareSccGraph(IReadOnlyDictionary<object, PathAwareSuccessorBucket> succIndex)
        {
            var adjacency = new Dictionary<object, List<object>>();
            var edgeCount = 0;

            void EnsureNode(object key)
            {
                if (!adjacency.ContainsKey(key))
                {
                    adjacency[key] = new List<object>();
                }
            }

            foreach (var (sourceKey, bucket) in succIndex)
            {
                EnsureNode(sourceKey);
                foreach (var target in bucket.Targets)
                {
                    var targetKey = target ?? NullFactIndexKey;
                    EnsureNode(targetKey);
                    adjacency[sourceKey].Add(targetKey);
                    edgeCount++;
                }
            }

            var index = 0;
            var indexByNode = new Dictionary<object, int>();
            var lowLink = new Dictionary<object, int>();
            var onStack = new HashSet<object>();
            var stack = new Stack<object>();
            var componentByNode = new Dictionary<object, int>();
            var componentSizes = new List<int>();
            var componentHasSelfLoop = new List<bool>();

            void StrongConnect(object node)
            {
                indexByNode[node] = index;
                lowLink[node] = index;
                index++;
                stack.Push(node);
                onStack.Add(node);

                foreach (var next in adjacency[node])
                {
                    if (!indexByNode.ContainsKey(next))
                    {
                        StrongConnect(next);
                        lowLink[node] = Math.Min(lowLink[node], lowLink[next]);
                    }
                    else if (onStack.Contains(next))
                    {
                        lowLink[node] = Math.Min(lowLink[node], indexByNode[next]);
                    }
                }

                if (lowLink[node] != indexByNode[node])
                {
                    return;
                }

                var componentId = componentSizes.Count;
                var size = 0;
                var hasSelfLoop = false;
                while (true)
                {
                    var member = stack.Pop();
                    onStack.Remove(member);
                    componentByNode[member] = componentId;
                    size++;
                    foreach (var next in adjacency[member])
                    {
                        if (Equals(next, member))
                        {
                            hasSelfLoop = true;
                            break;
                        }
                    }

                    if (Equals(member, node))
                    {
                        break;
                    }
                }

                componentSizes.Add(size);
                componentHasSelfLoop.Add(hasSelfLoop);
            }

            foreach (var node in adjacency.Keys.OrderBy(key => key, Comparer<object>.Create(CompareCacheSeedValues)))
            {
                if (!indexByNode.ContainsKey(node))
                {
                    StrongConnect(node);
                }
            }

            var condensedEdges = new HashSet<(int From, int To)>();
            foreach (var (sourceKey, targets) in adjacency)
            {
                var sourceComponent = componentByNode[sourceKey];
                foreach (var targetKey in targets)
                {
                    var targetComponent = componentByNode[targetKey];
                    if (sourceComponent != targetComponent)
                    {
                        condensedEdges.Add((sourceComponent, targetComponent));
                    }
                }
            }

            var cyclicComponentCount = 0;
            var largestComponentSize = 0;
            var largestCyclicComponentSize = 0;
            for (var componentId = 0; componentId < componentSizes.Count; componentId++)
            {
                var size = componentSizes[componentId];
                largestComponentSize = Math.Max(largestComponentSize, size);
                var cyclic = size > 1 || componentHasSelfLoop[componentId];
                if (!cyclic)
                {
                    continue;
                }

                cyclicComponentCount++;
                largestCyclicComponentSize = Math.Max(largestCyclicComponentSize, size);
            }

            return new PathAwareSccGraph(
                componentByNode,
                adjacency.Count,
                edgeCount,
                componentSizes.Count,
                cyclicComponentCount,
                largestComponentSize,
                largestCyclicComponentSize,
                condensedEdges.Count);
        }

        private static bool ShouldUseSccCondensedAdditiveMin(PathAwareSccGraph graph, int maxDepth)
        {
            const int MaxCyclicComponentSize = 128;
            return maxDepth > 0 &&
                graph.CyclicComponentCount > 0 &&
                graph.LargestCyclicComponentSize <= MaxCyclicComponentSize;
        }

        private const double SccCondensedProbeWinMargin = 0.50d;

        private static bool ResolveMeasuredSccCondensedAdditiveMinStrategy(
            QueryExecutionTrace? trace,
            PlanNode node,
            string strategyPrefix,
            PathAwareSccGraph graph,
            int maxDepth,
            string layeredProbePhase,
            IReadOnlyList<object?> orderedSeeds,
            Func<IReadOnlyList<object?>, TimeSpan> measureLayeredProbe,
            Func<IReadOnlyList<object?>, SccCondensedWeightedMinProbe> measureSccProbe)
        {
            if (!ShouldUseSccCondensedAdditiveMin(graph, maxDepth))
            {
                trace?.RecordStrategy(node, $"{strategyPrefix}SccCondensedRejectedStructural");
                return false;
            }

            if (orderedSeeds.Count == 0)
            {
                trace?.RecordStrategy(node, $"{strategyPrefix}SccCondensedRejectedNoSeeds");
                return false;
            }

            var sampleSeedCount = Math.Min(orderedSeeds.Count, 16);
            var sampleSeeds = orderedSeeds.Take(sampleSeedCount).ToList();
            var layeredProbe = measureLayeredProbe(sampleSeeds);
            trace?.RecordPhase(node, layeredProbePhase, layeredProbe);
            var sccProbe = measureSccProbe(sampleSeeds);
            trace?.RecordPhase(node, "scc_probe_condensed", sccProbe.Elapsed);
            trace?.RecordMetric(node, "scc_probe_local_states_explored", sccProbe.Stats.LocalStatesExplored);
            trace?.RecordMetric(node, "scc_probe_outer_dag_states_explored", sccProbe.Stats.OuterDagStatesExplored);
            trace?.RecordMetric(node, "scc_probe_queue_pops", sccProbe.Stats.QueuePops);

            if (layeredProbe <= TimeSpan.Zero || sccProbe.Elapsed <= TimeSpan.Zero)
            {
                trace?.RecordStrategy(node, $"{strategyPrefix}SccCondensedRejectedInvalidProbe");
                return false;
            }

            var useScc = sccProbe.Elapsed.Ticks < layeredProbe.Ticks * SccCondensedProbeWinMargin;
            trace?.RecordStrategy(node, useScc
                ? $"{strategyPrefix}SccCondensedSelectedMeasured"
                : $"{strategyPrefix}SccCondensedRejectedMeasured");
            return useScc;
        }

        private static string GetAdditiveMinLayeredStrategy(string strategyPrefix, AdditiveMinStepSafety stepSafety) =>
            stepSafety == AdditiveMinStepSafety.StrictlyPositive
                ? $"{strategyPrefix}PositiveAdditiveLayered"
                : $"{strategyPrefix}NonNegativeAdditiveLayered";

        private static string GetAdditiveMinLayeredProbePhase(AdditiveMinStepSafety stepSafety) =>
            stepSafety == AdditiveMinStepSafety.StrictlyPositive
                ? "scc_probe_positive_layered"
                : "scc_probe_nonnegative_layered";

        private static string GetAdditiveMinLayeredSolvePhase(AdditiveMinStepSafety stepSafety) =>
            stepSafety == AdditiveMinStepSafety.StrictlyPositive
                ? "positive_min_layered_solve"
                : "nonnegative_min_layered_solve";

        private static void RecordPathAwareSccGraphMetrics(QueryExecutionTrace? trace, PlanNode node, PathAwareSccGraph graph)
        {
            trace?.RecordMetric(node, "scc_node_count", graph.NodeCount);
            trace?.RecordMetric(node, "scc_edge_count", graph.EdgeCount);
            trace?.RecordMetric(node, "scc_count", graph.ComponentCount);
            trace?.RecordMetric(node, "scc_cyclic_count", graph.CyclicComponentCount);
            trace?.RecordMetric(node, "scc_largest_size", graph.LargestComponentSize);
            trace?.RecordMetric(node, "scc_largest_cyclic_size", graph.LargestCyclicComponentSize);
            trace?.RecordMetric(node, "scc_condensed_edge_count", graph.CondensedEdgeCount);
        }

        private static void RecordSccCondensedWeightedMinStats(
            QueryExecutionTrace? trace,
            PlanNode node,
            SccCondensedWeightedMinStats stats,
            int outputRows)
        {
            trace?.AddMetric(node, "scc_local_states_explored", stats.LocalStatesExplored);
            trace?.AddMetric(node, "scc_outer_dag_states_explored", stats.OuterDagStatesExplored);
            trace?.AddMetric(node, "scc_queue_pops", stats.QueuePops);
            trace?.AddMetric(node, "scc_output_rows", outputRows);
        }

        private static void RecordPathAwareMinFrontierMetrics(
            QueryExecutionTrace? trace,
            PlanNode node,
            PathAwareMinFrontierMetrics metrics)
        {
            trace?.AddMetric(node, "min_frontier_candidate_count", metrics.CandidateCount);
            trace?.AddMetric(node, "min_frontier_dominance_check_count", metrics.DominanceCheckCount);
            trace?.AddMetric(node, "min_frontier_dominance_candidate_check_count", metrics.DominanceCandidateCheckCount);
            trace?.AddMetric(node, "min_frontier_subset_check_count", metrics.SubsetCheckCount);
            trace?.AddMetric(node, "min_frontier_dominated_state_count", metrics.DominatedStateCount);
            trace?.AddMetric(node, "min_frontier_recorded_state_count", metrics.RecordedStateCount);
            trace?.AddMetric(node, "min_frontier_removed_state_count", metrics.RemovedStateCount);
            trace?.AddMetric(node, "min_frontier_target_bucket_count", metrics.TargetBucketCount);
            trace?.AddMetric(node, "min_frontier_bucket_count", metrics.BucketCount);
            trace?.AddMetric(node, "min_frontier_bucket_state_count", metrics.BucketStateCount);
            trace?.RecordMetric(node, "min_frontier_bucket_max_size", metrics.MaxBucketSize);
            trace?.RecordMetric(
                node,
                "min_frontier_bucket_avg_size",
                metrics.BucketCount == 0
                    ? 0d
                    : metrics.BucketStateCount / (double)metrics.BucketCount);
        }

        private IReadOnlyDictionary<object, Dictionary<object, int>> ExecuteSeedGroupedPathAwareDepthMinFallback(
            SeedGroupedPathAwareDepthMinNode closure,
            IReadOnlyList<object?> orderedUniqueSeeds,
            ISet<object> rootKeys,
            EvaluationContext context)
        {
            if (orderedUniqueSeeds.Count == 0)
            {
                return new Dictionary<object, Dictionary<object, int>>();
            }

            var seededClosure = new PathAwareTransitiveClosureNode(
                closure.EdgeRelation,
                closure.Predicate,
                closure.DirectSeedDepth,
                1,
                closure.MaxDepth,
                TableMode.Min);
            var seedParams = orderedUniqueSeeds.Select(seed => new object[] { seed! }).ToList();
            var rows = ExecuteSeededPathAwareTransitiveClosure(seededClosure, new[] { 0 }, seedParams, context).ToList();
            var seedMinima = new Dictionary<object, Dictionary<object, int>>();

            foreach (var seed in orderedUniqueSeeds)
            {
                var seedKey = seed ?? NullFactIndexKey;
                if (rootKeys.Contains(seedKey))
                {
                    seedMinima[seedKey] = new Dictionary<object, int>
                    {
                        [seedKey] = closure.DirectSeedDepth
                    };
                }
            }

            foreach (var row in rows)
            {
                if (row is null || row.Length < 3)
                {
                    continue;
                }

                var seed = row[0];
                var root = row[1];
                var rootKey = root ?? NullFactIndexKey;
                if (!rootKeys.Contains(rootKey))
                {
                    continue;
                }

                var seedKey = seed ?? NullFactIndexKey;
                if (!seedMinima.TryGetValue(seedKey, out var rootMins))
                {
                    rootMins = new Dictionary<object, int>();
                    seedMinima[seedKey] = rootMins;
                }

                var value = Convert.ToInt32(row[2], CultureInfo.InvariantCulture);
                if (!rootMins.TryGetValue(rootKey, out var current) || value < current)
                {
                    rootMins[rootKey] = value;
                }
            }

            return seedMinima;
        }

        private IEnumerable<object[]> ExecuteSeedGroupedPathAwareDepthMin(SeedGroupedPathAwareDepthMinNode closure, EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));

            var width = closure.Predicate.Arity;
            if (width < 3)
            {
                return Array.Empty<object[]>();
            }

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                trace?.RecordStrategy(closure, "SeedGroupedPathAwareDepthMin");

                if (context.SeedGroupedPathAwareDepthMinResults.TryGetValue(closure, out var cachedRows))
                {
                    trace?.RecordCacheLookup("SeedGroupedPathAwareDepthMin", $"{closure.Predicate.Name}/{closure.Predicate.Arity}", hit: true, built: false);
                    return cachedRows;
                }

                trace?.RecordCacheLookup("SeedGroupedPathAwareDepthMin", $"{closure.Predicate.Name}/{closure.Predicate.Arity}", hit: false, built: true);

                var edgeState = GetPathAwareEdgeState(closure.EdgeRelation, context, closure);
                var edgeRetentionSelection = GetCachedPathAwareEdgeRetentionSelection(closure.EdgeRelation, context);
                var succIndex = edgeState.Successors;
                var rootKeys = new HashSet<object>();
                var rootValues = new Dictionary<object, object?>();
                MeasurePhase(trace, closure, "load_roots", () =>
                {
                    foreach (var row in GetPathAwareSupportFactStream(closure.RootRelation, PathAwareSupportRelationAccessKind.Roots, context, closure))
                    {
                        if (row is null || row.Length == 0)
                        {
                            continue;
                        }

                        var root = row[0];
                        var rootKey = root ?? NullFactIndexKey;
                        if (rootKeys.Add(rootKey))
                        {
                            rootValues[rootKey] = root;
                        }
                    }
                });

                if (rootKeys.Count == 0)
                {
                    var empty = Array.Empty<object[]>();
                    context.SeedGroupedPathAwareDepthMinResults[closure] = empty;
                    return empty;
                }

                var groupedSeeds = new Dictionary<object, List<object?>>();
                var groupValues = new List<object?>();
                var uniqueSeeds = new HashSet<object?>();
                var orderedUniqueSeeds = new List<object?>();
                MeasurePhase(trace, closure, "load_seeds", () =>
                {
                    foreach (var row in GetPathAwareSupportFactStream(closure.SeedRelation, PathAwareSupportRelationAccessKind.Seeds, context, closure))
                    {
                        if (row is null || row.Length < 2)
                        {
                            continue;
                        }

                        var group = row[0];
                        var seed = row[1];
                        var groupKey = group ?? NullFactIndexKey;
                        if (!groupedSeeds.TryGetValue(groupKey, out var bucket))
                        {
                            bucket = new List<object?>();
                            groupedSeeds[groupKey] = bucket;
                            groupValues.Add(group);
                        }

                        bucket.Add(seed);
                        if (uniqueSeeds.Add(seed))
                        {
                            orderedUniqueSeeds.Add(seed);
                        }
                    }
                });

                if (groupValues.Count == 0)
                {
                    var empty = Array.Empty<object[]>();
                    context.SeedGroupedPathAwareDepthMinResults[closure] = empty;
                    return empty;
                }

                orderedUniqueSeeds.Sort(CompareCacheSeedValues);
                groupValues.Sort(CompareCacheSeedValues);

                Dictionary<object, Dictionary<object, int>> BuildCompactSeedDepthMins(IReadOnlyList<object?> seeds)
                {
                    var directSeedMins = new Dictionary<object, Dictionary<object, int>>();
                    foreach (var seed in seeds)
                    {
                        var mins = ComputePathAwareRootDepthMinsForSeed(seed, succIndex, rootKeys, closure.DirectSeedDepth, closure.MaxDepth);
                        if (mins.Count > 0)
                        {
                            directSeedMins[seed ?? NullFactIndexKey] = mins;
                        }
                    }

                    return directSeedMins;
                }

                TimeSpan MeasureCompactProbe(IReadOnlyList<object?> sampleSeeds) =>
                    MeasureElapsed(() => _ = BuildCompactSeedDepthMins(sampleSeeds));

                TimeSpan MeasureLegacyProbe(IReadOnlyList<object?> sampleSeeds) =>
                    MeasureElapsed(() => _ = ExecuteSeedGroupedPathAwareDepthMinFallback(closure, sampleSeeds, rootKeys, context));

                var materializationPlan = ResolveMaterializationPlan(
                    trace,
                    closure,
                    ToRelationRetentionSelection(edgeRetentionSelection),
                    () => ResolvePathAwareGroupedSummaryStrategy(
                        trace,
                        closure,
                        _pathAwareGroupedMinStrategy,
                        groupValues.Count,
                        orderedUniqueSeeds,
                        rootKeys.Count,
                        succIndex,
                        edgeRetentionSelection.Strategy,
                        MeasureCompactProbe,
                        MeasureLegacyProbe));
                RecordMaterializationPlanStrategy(trace, closure, "SeedGroupedPathAwareDepthMin", materializationPlan);
                var groupedSummarySelection = materializationPlan.GroupedSummary
                    ?? throw new InvalidOperationException("Expected grouped-summary selection for SeedGroupedPathAwareDepthMinNode.");
                RecordPathAwareGroupedSummaryStrategy(trace, closure, "SeedGroupedPathAwareDepthMin", groupedSummarySelection);

                IReadOnlyDictionary<object, Dictionary<object, int>> seedDepthMins = groupedSummarySelection.Strategy == PathAwareGroupedSummaryStrategy.LegacySeededRows
                    ? MeasurePhase(trace, closure, "build_legacy_seeded_rows", () =>
                        (IReadOnlyDictionary<object, Dictionary<object, int>>)ExecuteSeedGroupedPathAwareDepthMinFallback(closure, orderedUniqueSeeds, rootKeys, context))
                    : MeasurePhase(trace, closure, "build_compact_grouped", () =>
                        (IReadOnlyDictionary<object, Dictionary<object, int>>)BuildCompactSeedDepthMins(orderedUniqueSeeds));

                var rows = MeasurePhase(trace, closure, "group_reduce", () =>
                {
                    var builtRows = new List<object[]>();
                    foreach (var group in groupValues)
                    {
                        var groupKey = group ?? NullFactIndexKey;
                        if (!groupedSeeds.TryGetValue(groupKey, out var seedsForGroup))
                        {
                            continue;
                        }

                        Dictionary<object, int>? groupMins = null;
                        foreach (var seed in seedsForGroup)
                        {
                            if (!seedDepthMins.TryGetValue(seed ?? NullFactIndexKey, out var rootMins))
                            {
                                continue;
                            }

                            groupMins ??= new Dictionary<object, int>();
                            foreach (var entry in rootMins)
                            {
                                if (!groupMins.TryGetValue(entry.Key, out var current) || entry.Value < current)
                                {
                                    groupMins[entry.Key] = entry.Value;
                                }
                            }
                        }

                        if (groupMins is null)
                        {
                            continue;
                        }

                        var orderedRoots = groupMins.Keys.ToList();
                        orderedRoots.Sort((left, right) => CompareCacheSeedValues(rootValues[left], rootValues[right]));
                        foreach (var rootKey in orderedRoots)
                        {
                            builtRows.Add(new object[] { group!, rootValues[rootKey]!, groupMins[rootKey] });
                        }
                    }

                    return builtRows;
                });

                context.SeedGroupedPathAwareDepthMinResults[closure] = rows;
                return rows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private static Dictionary<object, int> ComputePathAwareRootDepthMinsForSeed(
            object? seed,
            IReadOnlyDictionary<object, PathAwareSuccessorBucket> succIndex,
            HashSet<object> rootKeys,
            int directSeedDepth,
            int maxDepth)
        {
            var mins = new Dictionary<object, int>();

            void Record(object? root, int depth)
            {
                var rootKey = root ?? NullFactIndexKey;
                if (!mins.TryGetValue(rootKey, out var current) || depth < current)
                {
                    mins[rootKey] = depth;
                }
            }

            var seedKey = seed ?? NullFactIndexKey;
            if (rootKeys.Contains(seedKey))
            {
                Record(seed, directSeedDepth);
            }

            var visited = new Dictionary<object, int> { [seedKey] = 0 };
            var queue = new Queue<(object? Node, int Depth)>();
            queue.Enqueue((seed, 0));

            while (queue.Count > 0)
            {
                var (current, depth) = queue.Dequeue();
                if (maxDepth > 0 && depth >= maxDepth)
                {
                    continue;
                }

                var currentKey = current ?? NullFactIndexKey;
                if (!succIndex.TryGetValue(currentKey, out var bucket))
                {
                    continue;
                }

                foreach (var next in bucket.Targets)
                {
                    var nextDepth = depth + 1;
                    var nextKey = next ?? NullFactIndexKey;
                    if (visited.TryGetValue(nextKey, out var bestDepth) && bestDepth <= nextDepth)
                    {
                        continue;
                    }

                    visited[nextKey] = nextDepth;
                    if (rootKeys.Contains(nextKey))
                    {
                        Record(next, directSeedDepth + nextDepth);
                    }

                    queue.Enqueue((next, nextDepth));
                }
            }

            return mins;
        }

        private IEnumerable<object[]> ExecuteSeedGroupedPathAwareAccumulationMin(SeedGroupedPathAwareAccumulationMinNode closure, EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));

            var width = closure.Predicate.Arity;
            if (width < 3)
            {
                return Array.Empty<object[]>();
            }

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                trace?.RecordStrategy(closure, "SeedGroupedPathAwareAccumulationMin");

                if (context.SeedGroupedPathAwareAccumulationMinResults.TryGetValue(closure, out var cachedRows))
                {
                    trace?.RecordCacheLookup("SeedGroupedPathAwareAccumulationMin", $"{closure.Predicate.Name}/{closure.Predicate.Arity}", hit: true, built: false);
                    return cachedRows;
                }

                trace?.RecordCacheLookup("SeedGroupedPathAwareAccumulationMin", $"{closure.Predicate.Name}/{closure.Predicate.Arity}", hit: false, built: true);

                var edgeState = GetPathAwareEdgeState(closure.EdgeRelation, context, closure);
                var edgeRetentionSelection = GetCachedPathAwareEdgeRetentionSelection(closure.EdgeRelation, context);
                var succIndex = edgeState.Successors;
                var rootKeys = new HashSet<object>();
                var rootValues = new Dictionary<object, object?>();
                MeasurePhase(trace, closure, "load_roots", () =>
                {
                    foreach (var row in GetPathAwareSupportFactStream(closure.RootRelation, PathAwareSupportRelationAccessKind.Roots, context, closure))
                    {
                        if (row is null || row.Length == 0)
                        {
                            continue;
                        }

                        var root = row[0];
                        var rootKey = root ?? NullFactIndexKey;
                        if (rootKeys.Add(rootKey))
                        {
                            rootValues[rootKey] = root;
                        }
                    }
                });

                if (rootKeys.Count == 0)
                {
                    var empty = Array.Empty<object[]>();
                    context.SeedGroupedPathAwareAccumulationMinResults[closure] = empty;
                    return empty;
                }

                var groupedSeeds = new Dictionary<object, List<object?>>();
                var groupValues = new List<object?>();
                var uniqueSeeds = new HashSet<object?>();
                var orderedUniqueSeeds = new List<object?>();
                MeasurePhase(trace, closure, "load_seeds", () =>
                {
                    foreach (var row in GetPathAwareSupportFactStream(closure.SeedRelation, PathAwareSupportRelationAccessKind.Seeds, context, closure))
                    {
                        if (row is null || row.Length < 2)
                        {
                            continue;
                        }

                        var group = row[0];
                        var seed = row[1];
                        var groupKey = group ?? NullFactIndexKey;
                        if (!groupedSeeds.TryGetValue(groupKey, out var bucket))
                        {
                            bucket = new List<object?>();
                            groupedSeeds[groupKey] = bucket;
                            groupValues.Add(group);
                        }

                        bucket.Add(seed);
                        if (uniqueSeeds.Add(seed))
                        {
                            orderedUniqueSeeds.Add(seed);
                        }
                    }
                });

                if (groupValues.Count == 0)
                {
                    var empty = Array.Empty<object[]>();
                    context.SeedGroupedPathAwareAccumulationMinResults[closure] = empty;
                    return empty;
                }

                orderedUniqueSeeds.Sort(CompareCacheSeedValues);
                groupValues.Sort(CompareCacheSeedValues);

                Dictionary<object, List<object[]>>? auxIndex = null;
                PositiveStepEvaluator? stepEvaluator = null;
                var additiveMinStepSafety = AdditiveMinStepSafety.StrictlyPositive;
                PathAwareSccGraph? sccGraph = null;
                var directSeedValue = Convert.ToDouble(closure.DirectSeedValue, CultureInfo.InvariantCulture);
                var additiveFastPathPrepared = false;
                var useAdditiveMinFastPath = false;
                var useSccCondensedMinFastPath = false;

                void EnsureAdditiveMinFastPathPrepared()
                {
                    if (additiveFastPathPrepared)
                    {
                        return;
                    }

                    additiveFastPathPrepared = true;
                    var auxRows = MeasurePhase(trace, closure, "load_auxiliary", () => GetClosureFactsList(closure.AuxiliaryRelation, ClosureRelationAccessKind.Support, context, closure));
                    auxIndex = MeasurePhase(trace, closure, "build_auxiliary_index", () => GetFactIndex(closure.AuxiliaryRelation, 0, auxRows, context));
                    useAdditiveMinFastPath = closure.MaxDepth > 0 &&
                        TryCreateNonNegativeAdditiveStepEvaluator(
                            closure.BaseExpression,
                            closure.RecursiveExpression,
                            succIndex,
                            auxIndex,
                            closure.PositiveStepProven,
                            out stepEvaluator,
                            out additiveMinStepSafety);
                    if (useAdditiveMinFastPath)
                    {
                        sccGraph = MeasurePhase(trace, closure, "scc_condense_graph", () => BuildPathAwareSccGraph(succIndex));
                        RecordPathAwareSccGraphMetrics(trace, closure, sccGraph);
                        useSccCondensedMinFastPath = ResolveMeasuredSccCondensedAdditiveMinStrategy(
                            trace,
                            closure,
                            "SeedGroupedPathAwareAccumulationMin",
                            sccGraph,
                            closure.MaxDepth,
                            GetAdditiveMinLayeredProbePhase(additiveMinStepSafety),
                            orderedUniqueSeeds,
                            sampleSeeds => MeasureElapsed(() =>
                            {
                                foreach (var seed in sampleSeeds)
                                {
                                    _ = ComputePositiveMinRootAccumulationsForSeed(
                                        seed,
                                        succIndex,
                                        auxIndex,
                                        rootKeys,
                                        stepEvaluator!,
                                        directSeedValue,
                                        closure.MaxDepth);
                                }
                            }),
                            sampleSeeds =>
                            {
                                long localStates = 0;
                                long outerDagStates = 0;
                                long queuePops = 0;
                                var elapsed = MeasureElapsed(() =>
                                {
                                    foreach (var seed in sampleSeeds)
                                    {
                                        _ = ComputeSccCondensedPositiveMinAccumulationsForSeed(
                                            seed,
                                            succIndex,
                                            auxIndex,
                                            rootKeys,
                                            sccGraph,
                                            stepEvaluator!,
                                            closure.MaxDepth,
                                            directSeedValue,
                                            out var stats);
                                        localStates += stats.LocalStatesExplored;
                                        outerDagStates += stats.OuterDagStatesExplored;
                                        queuePops += stats.QueuePops;
                                    }
                                });
                                return new SccCondensedWeightedMinProbe(
                                    elapsed,
                                    new SccCondensedWeightedMinStats(localStates, outerDagStates, queuePops));
                            });
                    }
                }

                Dictionary<object, Dictionary<object, object>> BuildCompactSeedMinima(IReadOnlyList<object?> seeds, bool recordMetrics)
                {
                    EnsureAdditiveMinFastPathPrepared();
                    if (!useAdditiveMinFastPath || auxIndex is null || stepEvaluator is null)
                    {
                        return new Dictionary<object, Dictionary<object, object>>();
                    }

                    if (useSccCondensedMinFastPath && sccGraph is not null)
                    {
                        if (recordMetrics)
                        {
                            trace?.RecordStrategy(closure, "SeedGroupedPathAwareAccumulationMinSccCondensed");
                        }

                        long localStates = 0;
                        long outerDagStates = 0;
                        long queuePops = 0;
                        var sccSeedMinima = recordMetrics
                            ? MeasurePhase(trace, closure, "scc_condensed_solve", BuildSccMinima)
                            : BuildSccMinima();
                        if (recordMetrics)
                        {
                            RecordSccCondensedWeightedMinStats(
                                trace,
                                closure,
                                new SccCondensedWeightedMinStats(localStates, outerDagStates, queuePops),
                                sccSeedMinima.Sum(kvp => kvp.Value.Count));
                        }

                        return sccSeedMinima;

                        Dictionary<object, Dictionary<object, object>> BuildSccMinima()
                        {
                            var minima = new Dictionary<object, Dictionary<object, object>>();
                            foreach (var seed in seeds)
                            {
                                var mins = ComputeSccCondensedPositiveMinAccumulationsForSeed(
                                    seed,
                                    succIndex,
                                    auxIndex,
                                    rootKeys,
                                    sccGraph,
                                    stepEvaluator,
                                    closure.MaxDepth,
                                    directSeedValue,
                                    out var stats);
                                localStates += stats.LocalStatesExplored;
                                outerDagStates += stats.OuterDagStatesExplored;
                                queuePops += stats.QueuePops;
                                if (mins.Count > 0)
                                {
                                    minima[seed ?? NullFactIndexKey] = mins.ToDictionary(kvp => kvp.Key, kvp => (object)kvp.Value);
                                }
                            }

                            return minima;
                        }
                    }

                    if (recordMetrics)
                    {
                        trace?.RecordStrategy(closure, GetAdditiveMinLayeredStrategy(
                            "SeedGroupedPathAwareAccumulationMin",
                            additiveMinStepSafety));
                    }

                    var fastSeedMinima = new Dictionary<object, Dictionary<object, object>>();
                    void BuildLayeredMinima()
                    {
                        foreach (var seed in seeds)
                        {
                            var mins = ComputePositiveMinRootAccumulationsForSeed(seed, succIndex, auxIndex, rootKeys, stepEvaluator, directSeedValue, closure.MaxDepth);
                            if (mins.Count > 0)
                            {
                                fastSeedMinima[seed ?? NullFactIndexKey] = mins.ToDictionary(kvp => kvp.Key, kvp => (object)kvp.Value);
                            }
                        }
                    }

                    if (recordMetrics)
                    {
                        MeasurePhase(trace, closure, GetAdditiveMinLayeredSolvePhase(additiveMinStepSafety), BuildLayeredMinima);
                    }
                    else
                    {
                        BuildLayeredMinima();
                    }

                    return fastSeedMinima;
                }

                TimeSpan MeasureCompactProbe(IReadOnlyList<object?> sampleSeeds) =>
                    MeasureElapsed(() =>
                    {
                        EnsureAdditiveMinFastPathPrepared();
                        if (!useAdditiveMinFastPath || auxIndex is null || stepEvaluator is null)
                        {
                            return;
                        }

                        _ = BuildCompactSeedMinima(sampleSeeds, recordMetrics: false);
                    });

                TimeSpan MeasureLegacyProbe(IReadOnlyList<object?> sampleSeeds) =>
                    MeasureElapsed(() => _ = ExecuteSeedGroupedPathAwareAccumulationMinFallback(closure, sampleSeeds, rootKeys, context));

                var materializationPlan = ResolveMaterializationPlan(
                    trace,
                    closure,
                    ToRelationRetentionSelection(edgeRetentionSelection),
                    () => ResolvePathAwareGroupedSummaryStrategy(
                        trace,
                        closure,
                        _pathAwareGroupedMinStrategy,
                        groupValues.Count,
                        orderedUniqueSeeds,
                        rootKeys.Count,
                        succIndex,
                        edgeRetentionSelection.Strategy,
                        MeasureCompactProbe,
                        MeasureLegacyProbe));
                RecordMaterializationPlanStrategy(trace, closure, "SeedGroupedPathAwareAccumulationMin", materializationPlan);
                var groupedSummarySelection = materializationPlan.GroupedSummary
                    ?? throw new InvalidOperationException("Expected grouped-summary selection for SeedGroupedPathAwareAccumulationMinNode.");
                RecordPathAwareGroupedSummaryStrategy(trace, closure, "SeedGroupedPathAwareAccumulationMin", groupedSummarySelection);

                IReadOnlyDictionary<object, Dictionary<object, object>> seedMinima;
                if (groupedSummarySelection.Strategy == PathAwareGroupedSummaryStrategy.LegacySeededRows)
                {
                    seedMinima = MeasurePhase(trace, closure, "build_legacy_seeded_rows", () =>
                        (IReadOnlyDictionary<object, Dictionary<object, object>>)ExecuteSeedGroupedPathAwareAccumulationMinFallback(closure, orderedUniqueSeeds, rootKeys, context));
                }
                else
                {
                    EnsureAdditiveMinFastPathPrepared();
                    if (useAdditiveMinFastPath && auxIndex is not null && stepEvaluator is not null)
                    {
                        seedMinima = MeasurePhase(trace, closure, "build_compact_grouped", () =>
                            (IReadOnlyDictionary<object, Dictionary<object, object>>)BuildCompactSeedMinima(orderedUniqueSeeds, recordMetrics: true));
                    }
                    else
                    {
                        trace?.RecordStrategy(closure, "SeedGroupedPathAwareAccumulationMinLegacySeededRowsNoAdditiveFastPath");
                        trace?.RecordStrategy(closure, "SeedGroupedPathAwareAccumulationMinMaterializationPlanFallbackLegacyNoAdditiveFastPath");
                        seedMinima = MeasurePhase(trace, closure, "build_legacy_seeded_rows", () =>
                            (IReadOnlyDictionary<object, Dictionary<object, object>>)ExecuteSeedGroupedPathAwareAccumulationMinFallback(closure, orderedUniqueSeeds, rootKeys, context));
                    }
                }

                var rows = MeasurePhase(trace, closure, "group_reduce", () =>
                {
                    var builtRows = new List<object[]>();
                    foreach (var group in groupValues)
                    {
                        var groupKey = group ?? NullFactIndexKey;
                        if (!groupedSeeds.TryGetValue(groupKey, out var seedsForGroup))
                        {
                            continue;
                        }

                        Dictionary<object, object>? groupMins = null;
                        foreach (var seed in seedsForGroup)
                        {
                            if (!seedMinima.TryGetValue(seed ?? NullFactIndexKey, out var rootMins))
                            {
                                continue;
                            }

                            groupMins ??= new Dictionary<object, object>();
                            foreach (var entry in rootMins)
                            {
                                if (!groupMins.TryGetValue(entry.Key, out var current) || CompareValues(entry.Value, current) < 0)
                                {
                                    groupMins[entry.Key] = entry.Value;
                                }
                            }
                        }

                        if (groupMins is null)
                        {
                            continue;
                        }

                        var orderedRoots = groupMins.Keys.ToList();
                        orderedRoots.Sort((left, right) => CompareCacheSeedValues(rootValues[left], rootValues[right]));
                        foreach (var rootKey in orderedRoots)
                        {
                            builtRows.Add(new object[] { group!, rootValues[rootKey]!, groupMins[rootKey] });
                        }
                    }

                    return builtRows;
                });

                context.SeedGroupedPathAwareAccumulationMinResults[closure] = rows;
                return rows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private IReadOnlyDictionary<object, Dictionary<object, object>> ExecuteSeedGroupedPathAwareAccumulationMinFallback(
            SeedGroupedPathAwareAccumulationMinNode closure,
            IReadOnlyList<object?> orderedUniqueSeeds,
            ISet<object> rootKeys,
            EvaluationContext context)
        {
            if (orderedUniqueSeeds.Count == 0)
            {
                return new Dictionary<object, Dictionary<object, object>>();
            }

            var seededClosure = new PathAwareAccumulationNode(
                closure.EdgeRelation,
                closure.Predicate,
                closure.AuxiliaryRelation,
                closure.BaseExpression,
                closure.RecursiveExpression,
                closure.MaxDepth,
                TableMode.Min,
                closure.PositiveStepProven);
            var seedParams = orderedUniqueSeeds.Select(seed => new object[] { seed! }).ToList();
            var rows = ExecuteSeededPathAwareAccumulation(seededClosure, new[] { 0 }, seedParams, context).ToList();
            var seedMinima = new Dictionary<object, Dictionary<object, object>>();

            foreach (var seed in orderedUniqueSeeds)
            {
                var seedKey = seed ?? NullFactIndexKey;
                if (rootKeys.Contains(seedKey))
                {
                    seedMinima[seedKey] = new Dictionary<object, object>
                    {
                        [seedKey] = closure.DirectSeedValue
                    };
                }
            }

            foreach (var row in rows)
            {
                if (row is null || row.Length < 3)
                {
                    continue;
                }

                var seed = row[0];
                var root = row[1];
                var rootKey = root ?? NullFactIndexKey;
                if (!rootKeys.Contains(rootKey))
                {
                    continue;
                }

                var seedKey = seed ?? NullFactIndexKey;
                if (!seedMinima.TryGetValue(seedKey, out var rootMins))
                {
                    rootMins = new Dictionary<object, object>();
                    seedMinima[seedKey] = rootMins;
                }

                var value = row[2];
                if (!rootMins.TryGetValue(rootKey, out var current) || CompareValues(value, current) < 0)
                {
                    rootMins[rootKey] = value;
                }
            }

            return seedMinima;
        }

        private static Dictionary<object, double> ComputePositiveMinRootAccumulationsForSeed(
            object? seed,
            IReadOnlyDictionary<object, PathAwareSuccessorBucket> succIndex,
            IReadOnlyDictionary<object, List<object[]>> auxIndex,
            ISet<object> rootKeys,
            PositiveStepEvaluator stepEvaluator,
            double directSeedValue,
            int maxDepth)
        {
            var mins = new Dictionary<object, double>();
            var seedKey = seed ?? NullFactIndexKey;
            if (rootKeys.Contains(seedKey))
            {
                mins[seedKey] = directSeedValue;
            }

            var depthCosts = new List<Dictionary<object?, double>>(maxDepth + 1)
            {
                new Dictionary<object?, double> { [seed] = 0d }
            };

            for (var depth = 0; depth < maxDepth && depth < depthCosts.Count; depth++)
            {
                var currentLayer = depthCosts[depth];
                if (currentLayer.Count == 0)
                {
                    continue;
                }

                foreach (var (current, currentCost) in currentLayer)
                {
                    var currentKey = current ?? NullFactIndexKey;
                    if (!succIndex.TryGetValue(currentKey, out var edgeBucket) || !auxIndex.TryGetValue(currentKey, out var auxBucket))
                    {
                        continue;
                    }

                    var nextDepth = depth + 1;
                    while (depthCosts.Count <= nextDepth)
                    {
                        depthCosts.Add(new Dictionary<object?, double>());
                    }

                    var nextLayer = depthCosts[nextDepth];
                    for (var edgeIndex = edgeBucket.Targets.Count - 1; edgeIndex >= 0; edgeIndex--)
                    {
                        var next = edgeBucket.Targets[edgeIndex];
                        var nextKey = next ?? NullFactIndexKey;
                        if (Equals(nextKey, seedKey))
                        {
                            continue;
                        }

                        for (var auxIndexPos = auxBucket.Count - 1; auxIndexPos >= 0; auxIndexPos--)
                        {
                            var auxRow = auxBucket[auxIndexPos];
                            if (auxRow is null || auxRow.Length < 2)
                            {
                                continue;
                            }

                            var step = stepEvaluator(current!, next!, auxRow[1]!);
                            var nextCost = currentCost + step;
                            if (!nextLayer.TryGetValue(next, out var existingCost) || nextCost < existingCost)
                            {
                                nextLayer[next] = nextCost;
                            }

                            if (!rootKeys.Contains(nextKey))
                            {
                                continue;
                            }

                            if (!mins.TryGetValue(nextKey, out var currentMin) || nextCost < currentMin)
                            {
                                mins[nextKey] = nextCost;
                            }
                        }
                    }
                }
            }

            return mins;
        }

        private static Dictionary<object, double> ComputeSccCondensedPositiveMinAccumulationsForSeed(
            object? seed,
            IReadOnlyDictionary<object, PathAwareSuccessorBucket> succIndex,
            IReadOnlyDictionary<object, List<object[]>> auxIndex,
            ISet<object>? rootKeys,
            PathAwareSccGraph sccGraph,
            PositiveStepEvaluator stepEvaluator,
            int maxDepth,
            double? directSeedValue,
            out SccCondensedWeightedMinStats stats)
        {
            var mins = new Dictionary<object, double>();
            var seedKey = seed ?? NullFactIndexKey;
            if (directSeedValue is { } directValue && rootKeys is not null && rootKeys.Contains(seedKey))
            {
                mins[seedKey] = directValue;
            }

            var bestByDepth = new Dictionary<(object Key, int Depth), double>
            {
                [(seedKey, 0)] = 0d
            };
            var queue = new PriorityQueue<(object? Node, object Key, int Depth, double Cost), double>();
            queue.Enqueue((seed, seedKey, 0, 0d), 0d);

            long localStates = 0;
            long outerDagStates = 0;
            long queuePops = 0;

            while (queue.Count > 0)
            {
                var (current, currentKey, depth, currentCost) = queue.Dequeue();
                queuePops++;
                if (!bestByDepth.TryGetValue((currentKey, depth), out var recordedCost) || currentCost > recordedCost)
                {
                    continue;
                }

                if (depth >= maxDepth)
                {
                    continue;
                }

                if (!succIndex.TryGetValue(currentKey, out var edgeBucket) || !auxIndex.TryGetValue(currentKey, out var auxBucket))
                {
                    continue;
                }

                sccGraph.ComponentByNode.TryGetValue(currentKey, out var currentComponent);
                var nextDepth = depth + 1;
                for (var edgeIndex = edgeBucket.Targets.Count - 1; edgeIndex >= 0; edgeIndex--)
                {
                    var next = edgeBucket.Targets[edgeIndex];
                    var nextKey = next ?? NullFactIndexKey;
                    if (Equals(nextKey, seedKey))
                    {
                        continue;
                    }

                    sccGraph.ComponentByNode.TryGetValue(nextKey, out var nextComponent);
                    for (var auxIndexPos = auxBucket.Count - 1; auxIndexPos >= 0; auxIndexPos--)
                    {
                        var auxRow = auxBucket[auxIndexPos];
                        if (auxRow is null || auxRow.Length < 2)
                        {
                            continue;
                        }

                        var step = stepEvaluator(current!, next!, auxRow[1]!);
                        var nextCost = currentCost + step;
                        var stateKey = (nextKey, nextDepth);
                        if (bestByDepth.TryGetValue(stateKey, out var existingCost) && existingCost <= nextCost)
                        {
                            continue;
                        }

                        bestByDepth[stateKey] = nextCost;
                        queue.Enqueue((next, nextKey, nextDepth, nextCost), nextCost);
                        if (currentComponent == nextComponent)
                        {
                            localStates++;
                        }
                        else
                        {
                            outerDagStates++;
                        }

                        if (rootKeys is not null && !rootKeys.Contains(nextKey))
                        {
                            continue;
                        }

                        if (!mins.TryGetValue(nextKey, out var currentMin) || nextCost < currentMin)
                        {
                            mins[nextKey] = nextCost;
                        }
                    }
                }
            }

            stats = new SccCondensedWeightedMinStats(localStates, outerDagStates, queuePops);
            return mins;
        }

        private IEnumerable<object[]> ExecutePathAwareTransitiveClosure(PathAwareTransitiveClosureNode closure, EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                trace?.RecordStrategy(closure, "PathAwareTransitiveClosure");

                var edgeState = GetPathAwareEdgeState(closure.EdgeRelation, context, closure);
                var totalRows = new List<object[]>();
                foreach (var seed in edgeState.Seeds)
                {
                    AppendPathAwareRowsForSeed(seed, edgeState.Successors, closure.BaseDepth, closure.DepthIncrement, totalRows, closure.AccumulatorMode, closure.MaxDepth);
                }

                return totalRows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private IEnumerable<object[]> ExecuteSeededPathAwareTransitiveClosure(
            PathAwareTransitiveClosureNode closure,
            IReadOnlyList<int> inputPositions,
            IReadOnlyList<object[]> parameters,
            EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));
            if (inputPositions is null) throw new ArgumentNullException(nameof(inputPositions));
            if (parameters is null) throw new ArgumentNullException(nameof(parameters));

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                trace?.RecordStrategy(closure, "PathAwareTransitiveClosureSeeded");

                var edgeState = GetPathAwareEdgeState(closure.EdgeRelation, context, closure);
                var seeds = new List<object?>();
                var seenSeeds = new HashSet<object?>();

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

                    if (seenSeeds.Add(seed))
                    {
                        seeds.Add(seed);
                    }
                }

                seeds.Sort(CompareCacheSeedValues);
                var totalRows = new List<object[]>();
                foreach (var seed in seeds)
                {
                    AppendPathAwareRowsForSeed(seed, edgeState.Successors, closure.BaseDepth, closure.DepthIncrement, totalRows, closure.AccumulatorMode, closure.MaxDepth);
                }

                return totalRows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private static void AppendPathAwareRowsForSeed(
            object? seed,
            IReadOnlyDictionary<object, PathAwareSuccessorBucket> succIndex,
            int baseDepth,
            int depthIncrement,
            ICollection<object[]> output,
            TableMode accumulatorMode,
            int maxDepth = 0)
        {
            var preserveAllPaths = accumulatorMode is TableMode.All or TableMode.Sum or TableMode.Count;
            var bestKnown = preserveAllPaths ? null : new Dictionary<object?, int>();
            var stack = new Stack<(object? Node, int Depth, HashSet<object?> Visited)>();
            stack.Push((seed, 0, new HashSet<object?> { seed }));

            while (stack.Count > 0)
            {
                var (current, depth, visited) = stack.Pop();
                var lookupKey = current ?? NullFactIndexKey;
                if (!succIndex.TryGetValue(lookupKey, out var bucket))
                {
                    continue;
                }

                for (var i = bucket.Targets.Count - 1; i >= 0; i--)
                {
                    var next = bucket.Targets[i];
                    if (visited.Contains(next))
                    {
                        continue;
                    }

                    var nextDepth = depth == 0
                        ? baseDepth
                        : checked(depth + depthIncrement);

                    if (maxDepth > 0 && nextDepth > maxDepth)
                    {
                        continue;
                    }

                    if (bestKnown is not null)
                    {
                        if (bestKnown.TryGetValue(next, out var bestDepth))
                        {
                            switch (accumulatorMode)
                            {
                                case TableMode.Min:
                                    if (nextDepth >= bestDepth)
                                    {
                                        continue;
                                    }
                                    break;
                                case TableMode.Max:
                                    if (nextDepth <= bestDepth)
                                    {
                                        continue;
                                    }
                                    break;
                                case TableMode.First:
                                    continue;
                            }
                        }

                        bestKnown[next] = nextDepth;
                    }
                    else
                    {
                        output.Add(new object[] { seed!, next!, nextDepth });
                    }

                    var nextVisited = new HashSet<object?>(visited) { next };
                    stack.Push((next, nextDepth, nextVisited));
                }
            }

            if (bestKnown is not null)
            {
                var bestRows = new List<KeyValuePair<object?, int>>(bestKnown);
                bestRows.Sort((left, right) => CompareCacheSeedValues(left.Key, right.Key));
                foreach (var (target, depth) in bestRows)
                {
                    output.Add(new object[] { seed!, target!, depth });
                }
            }
        }

        private IEnumerable<object[]> ExecutePathAwareAccumulation(PathAwareAccumulationNode closure, EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                trace?.RecordStrategy(closure, "PathAwareAccumulation");

                var edgeState = GetPathAwareEdgeState(closure.EdgeRelation, context, closure);
                var succIndex = edgeState.Successors;
                var auxRows = GetClosureFactsList(closure.AuxiliaryRelation, ClosureRelationAccessKind.Support, context, closure);
                var auxIndex = GetFactIndex(closure.AuxiliaryRelation, 0, auxRows, context);
                PositiveStepEvaluator? stepEvaluator = null;
                var additiveMinStepSafety = AdditiveMinStepSafety.StrictlyPositive;
                var useAdditiveMinFastPath = closure.AccumulatorMode == TableMode.Min &&
                    closure.MaxDepth > 0 &&
                    TryCreateNonNegativeAdditiveStepEvaluator(
                        closure.BaseExpression,
                        closure.RecursiveExpression,
                        succIndex,
                        auxIndex,
                        closure.PositiveStepProven,
                        out stepEvaluator,
                        out additiveMinStepSafety);
                PathAwareSccGraph? sccGraph = null;
                var useSccCondensedMinFastPath = false;
                if (useAdditiveMinFastPath)
                {
                    sccGraph = MeasurePhase(trace, closure, "scc_condense_graph", () => BuildPathAwareSccGraph(succIndex));
                    RecordPathAwareSccGraphMetrics(trace, closure, sccGraph);
                    useSccCondensedMinFastPath = ResolveMeasuredSccCondensedAdditiveMinStrategy(
                        trace,
                        closure,
                        "PathAwareAccumulationMin",
                        sccGraph,
                        closure.MaxDepth,
                        GetAdditiveMinLayeredProbePhase(additiveMinStepSafety),
                        edgeState.Seeds,
                        sampleSeeds => MeasureElapsed(() =>
                        {
                            var probeRows = new List<object[]>();
                            foreach (var seed in sampleSeeds)
                            {
                                AppendPositiveMinAccumulationRowsForSeed(seed, succIndex, auxIndex, stepEvaluator!, probeRows, closure.MaxDepth);
                            }
                        }),
                        sampleSeeds =>
                        {
                            long localStates = 0;
                            long outerDagStates = 0;
                            long queuePops = 0;
                            var elapsed = MeasureElapsed(() =>
                            {
                                var probeRows = new List<object[]>();
                                foreach (var seed in sampleSeeds)
                                {
                                    var stats = AppendSccCondensedPositiveMinAccumulationRowsForSeed(seed, succIndex, auxIndex, sccGraph, stepEvaluator!, probeRows, closure.MaxDepth);
                                    localStates += stats.LocalStatesExplored;
                                    outerDagStates += stats.OuterDagStatesExplored;
                                    queuePops += stats.QueuePops;
                                }
                            });
                            return new SccCondensedWeightedMinProbe(
                                elapsed,
                                new SccCondensedWeightedMinStats(localStates, outerDagStates, queuePops));
                        });
                    trace?.RecordStrategy(closure, useSccCondensedMinFastPath
                        ? "PathAwareAccumulationMinSccCondensed"
                        : GetAdditiveMinLayeredStrategy("PathAwareAccumulationMin", additiveMinStepSafety));
                }
                else if (closure.AccumulatorMode == TableMode.Min)
                {
                    trace?.RecordStrategy(closure, "PathAwareAccumulationMinFrontierFallback");
                }

                var totalRows = new List<object[]>();
                long localStates = 0;
                long outerDagStates = 0;
                long queuePops = 0;
                var frontierMetrics = useAdditiveMinFastPath
                    ? null
                    : new PathAwareMinFrontierMetrics();
                void BuildAccumulationRows()
                {
                    foreach (var seed in edgeState.Seeds)
                    {
                        if (useSccCondensedMinFastPath && sccGraph is not null)
                        {
                            var stats = AppendSccCondensedPositiveMinAccumulationRowsForSeed(seed, succIndex, auxIndex, sccGraph, stepEvaluator!, totalRows, closure.MaxDepth);
                            localStates += stats.LocalStatesExplored;
                            outerDagStates += stats.OuterDagStatesExplored;
                            queuePops += stats.QueuePops;
                        }
                        else if (useAdditiveMinFastPath)
                        {
                            AppendPositiveMinAccumulationRowsForSeed(seed, succIndex, auxIndex, stepEvaluator!, totalRows, closure.MaxDepth);
                        }
                        else
                        {
                            AppendPathAwareAccumulationRowsForSeed(seed, succIndex, auxIndex, closure.BaseExpression, closure.RecursiveExpression, totalRows, closure.AccumulatorMode, closure.MaxDepth, frontierMetrics);
                        }
                    }
                }

                if (useSccCondensedMinFastPath)
                {
                    MeasurePhase(trace, closure, "scc_condensed_solve", BuildAccumulationRows);
                }
                else if (useAdditiveMinFastPath)
                {
                    MeasurePhase(trace, closure, GetAdditiveMinLayeredSolvePhase(additiveMinStepSafety), BuildAccumulationRows);
                }
                else
                {
                    BuildAccumulationRows();
                }

                if (useSccCondensedMinFastPath)
                {
                    RecordSccCondensedWeightedMinStats(
                        trace,
                        closure,
                        new SccCondensedWeightedMinStats(localStates, outerDagStates, queuePops),
                        totalRows.Count);
                }
                else if (!useAdditiveMinFastPath && closure.AccumulatorMode == TableMode.Min && frontierMetrics is not null)
                {
                    RecordPathAwareMinFrontierMetrics(trace, closure, frontierMetrics);
                }

                return totalRows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private IEnumerable<object[]> ExecuteSeededPathAwareAccumulation(
            PathAwareAccumulationNode closure,
            IReadOnlyList<int> inputPositions,
            IReadOnlyList<object[]> parameters,
            EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));
            if (inputPositions is null) throw new ArgumentNullException(nameof(inputPositions));
            if (parameters is null) throw new ArgumentNullException(nameof(parameters));

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                trace?.RecordStrategy(closure, "PathAwareAccumulationSeeded");

                var edgeState = GetPathAwareEdgeState(closure.EdgeRelation, context, closure);
                var succIndex = edgeState.Successors;
                var auxRows = GetClosureFactsList(closure.AuxiliaryRelation, ClosureRelationAccessKind.Support, context, closure);
                var auxIndex = GetFactIndex(closure.AuxiliaryRelation, 0, auxRows, context);
                var seeds = new List<object?>();
                var seenSeeds = new HashSet<object?>();
                PositiveStepEvaluator? stepEvaluator = null;
                var additiveMinStepSafety = AdditiveMinStepSafety.StrictlyPositive;
                var useAdditiveMinFastPath = closure.AccumulatorMode == TableMode.Min &&
                    closure.MaxDepth > 0 &&
                    TryCreateNonNegativeAdditiveStepEvaluator(
                        closure.BaseExpression,
                        closure.RecursiveExpression,
                        succIndex,
                        auxIndex,
                        closure.PositiveStepProven,
                        out stepEvaluator,
                        out additiveMinStepSafety);
                PathAwareSccGraph? sccGraph = null;
                var useSccCondensedMinFastPath = false;
                if (useAdditiveMinFastPath)
                {
                    sccGraph = MeasurePhase(trace, closure, "scc_condense_graph", () => BuildPathAwareSccGraph(succIndex));
                    RecordPathAwareSccGraphMetrics(trace, closure, sccGraph);
                }

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

                    if (seenSeeds.Add(seed))
                    {
                        seeds.Add(seed);
                    }
                }

                seeds.Sort(CompareCacheSeedValues);
                if (useAdditiveMinFastPath && sccGraph is not null)
                {
                    useSccCondensedMinFastPath = ResolveMeasuredSccCondensedAdditiveMinStrategy(
                        trace,
                        closure,
                        "PathAwareAccumulationSeededMin",
                        sccGraph,
                        closure.MaxDepth,
                        GetAdditiveMinLayeredProbePhase(additiveMinStepSafety),
                        seeds,
                        sampleSeeds => MeasureElapsed(() =>
                        {
                            var probeRows = new List<object[]>();
                            foreach (var seed in sampleSeeds)
                            {
                                AppendPositiveMinAccumulationRowsForSeed(seed, succIndex, auxIndex, stepEvaluator!, probeRows, closure.MaxDepth);
                            }
                        }),
                        sampleSeeds =>
                        {
                            long localStates = 0;
                            long outerDagStates = 0;
                            long queuePops = 0;
                            var elapsed = MeasureElapsed(() =>
                            {
                                var probeRows = new List<object[]>();
                                foreach (var seed in sampleSeeds)
                                {
                                    var stats = AppendSccCondensedPositiveMinAccumulationRowsForSeed(seed, succIndex, auxIndex, sccGraph, stepEvaluator!, probeRows, closure.MaxDepth);
                                    localStates += stats.LocalStatesExplored;
                                    outerDagStates += stats.OuterDagStatesExplored;
                                    queuePops += stats.QueuePops;
                                }
                            });
                            return new SccCondensedWeightedMinProbe(
                                elapsed,
                                new SccCondensedWeightedMinStats(localStates, outerDagStates, queuePops));
                        });
                    trace?.RecordStrategy(closure, useSccCondensedMinFastPath
                        ? "PathAwareAccumulationSeededMinSccCondensed"
                        : GetAdditiveMinLayeredStrategy("PathAwareAccumulationSeededMin", additiveMinStepSafety));
                }
                else if (closure.AccumulatorMode == TableMode.Min)
                {
                    trace?.RecordStrategy(closure, "PathAwareAccumulationSeededMinFrontierFallback");
                }

                var totalRows = new List<object[]>();
                long localStates = 0;
                long outerDagStates = 0;
                long queuePops = 0;
                var frontierMetrics = useAdditiveMinFastPath
                    ? null
                    : new PathAwareMinFrontierMetrics();
                void BuildAccumulationRows()
                {
                    foreach (var seed in seeds)
                    {
                        if (useSccCondensedMinFastPath && sccGraph is not null)
                        {
                            var stats = AppendSccCondensedPositiveMinAccumulationRowsForSeed(seed, succIndex, auxIndex, sccGraph, stepEvaluator!, totalRows, closure.MaxDepth);
                            localStates += stats.LocalStatesExplored;
                            outerDagStates += stats.OuterDagStatesExplored;
                            queuePops += stats.QueuePops;
                        }
                        else if (useAdditiveMinFastPath)
                        {
                            AppendPositiveMinAccumulationRowsForSeed(seed, succIndex, auxIndex, stepEvaluator!, totalRows, closure.MaxDepth);
                        }
                        else
                        {
                            AppendPathAwareAccumulationRowsForSeed(seed, succIndex, auxIndex, closure.BaseExpression, closure.RecursiveExpression, totalRows, closure.AccumulatorMode, closure.MaxDepth, frontierMetrics);
                        }
                    }
                }

                if (useSccCondensedMinFastPath)
                {
                    MeasurePhase(trace, closure, "scc_condensed_solve", BuildAccumulationRows);
                }
                else if (useAdditiveMinFastPath)
                {
                    MeasurePhase(trace, closure, GetAdditiveMinLayeredSolvePhase(additiveMinStepSafety), BuildAccumulationRows);
                }
                else
                {
                    BuildAccumulationRows();
                }

                if (useSccCondensedMinFastPath)
                {
                    RecordSccCondensedWeightedMinStats(
                        trace,
                        closure,
                        new SccCondensedWeightedMinStats(localStates, outerDagStates, queuePops),
                        totalRows.Count);
                }
                else if (!useAdditiveMinFastPath && closure.AccumulatorMode == TableMode.Min && frontierMetrics is not null)
                {
                    RecordPathAwareMinFrontierMetrics(trace, closure, frontierMetrics);
                }

                return totalRows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private delegate double PositiveStepEvaluator(object current, object next, object auxValue);

        // For non-negative additive steps, any repeated-node walk can have its
        // cycle removed without increasing cost. That lets us replace the
        // expensive visited-state frontier with dynamic programming over
        // (node, depth) for bounded Min queries.
        private void AppendPositiveMinAccumulationRowsForSeed(
            object? seed,
            IReadOnlyDictionary<object, PathAwareSuccessorBucket> succIndex,
            IReadOnlyDictionary<object, List<object[]>> auxIndex,
            PositiveStepEvaluator stepEvaluator,
            ICollection<object[]> output,
            int maxDepth = 0)
        {
            var effectiveMaxDepth = maxDepth > 0 ? maxDepth : int.MaxValue;
            var seedKey = seed ?? NullFactIndexKey;
            var depthCosts = new List<Dictionary<object?, double>>(effectiveMaxDepth == int.MaxValue ? 16 : effectiveMaxDepth + 1)
            {
                new Dictionary<object?, double> { [seed] = 0d }
            };
            var bestKnown = new Dictionary<object?, double>();

            for (var depth = 0; depth < effectiveMaxDepth && depth < depthCosts.Count; depth++)
            {
                var currentLayer = depthCosts[depth];
                if (currentLayer.Count == 0)
                {
                    continue;
                }

                foreach (var (current, currentCost) in currentLayer)
                {
                    var currentKey = current ?? NullFactIndexKey;
                    if (!succIndex.TryGetValue(currentKey, out var edgeBucket) || !auxIndex.TryGetValue(currentKey, out var auxBucket))
                    {
                        continue;
                    }

                    var nextDepth = depth + 1;
                    if (nextDepth > effectiveMaxDepth)
                    {
                        continue;
                    }

                    while (depthCosts.Count <= nextDepth)
                    {
                        depthCosts.Add(new Dictionary<object?, double>());
                    }

                    var nextLayer = depthCosts[nextDepth];
                    for (var edgeIndex = edgeBucket.Targets.Count - 1; edgeIndex >= 0; edgeIndex--)
                    {
                        var next = edgeBucket.Targets[edgeIndex];
                        if (Equals(next ?? NullFactIndexKey, seedKey))
                        {
                            continue;
                        }

                        for (var auxIndexPos = auxBucket.Count - 1; auxIndexPos >= 0; auxIndexPos--)
                        {
                            var auxRow = auxBucket[auxIndexPos];
                            if (auxRow is null || auxRow.Length < 2)
                            {
                                continue;
                            }

                            var step = stepEvaluator(current!, next!, auxRow[1]!);
                            var nextCost = currentCost + step;

                            if (!nextLayer.TryGetValue(next, out var existingCost) || nextCost < existingCost)
                            {
                                nextLayer[next] = nextCost;
                            }

                            if (!bestKnown.TryGetValue(next, out var bestCost) || nextCost < bestCost)
                            {
                                bestKnown[next] = nextCost;
                            }
                        }
                    }
                }
            }

            var bestRows = bestKnown
                .OrderBy(kvp => kvp.Key, Comparer<object?>.Create(CompareCacheSeedValues))
                .ToList();
            foreach (var (target, cost) in bestRows)
            {
                output.Add(new object[] { seed!, target!, cost });
            }
        }

        private static SccCondensedWeightedMinStats AppendSccCondensedPositiveMinAccumulationRowsForSeed(
            object? seed,
            IReadOnlyDictionary<object, PathAwareSuccessorBucket> succIndex,
            IReadOnlyDictionary<object, List<object[]>> auxIndex,
            PathAwareSccGraph sccGraph,
            PositiveStepEvaluator stepEvaluator,
            ICollection<object[]> output,
            int maxDepth)
        {
            var minima = ComputeSccCondensedPositiveMinAccumulationsForSeed(
                seed,
                succIndex,
                auxIndex,
                rootKeys: null,
                sccGraph,
                stepEvaluator,
                maxDepth,
                directSeedValue: null,
                out var stats);

            var bestRows = minima
                .OrderBy(kvp => kvp.Key, Comparer<object>.Create(CompareCacheSeedValues))
                .ToList();
            foreach (var (target, cost) in bestRows)
            {
                output.Add(new object[] { seed!, ReferenceEquals(target, NullFactIndexKey) ? null! : target, cost });
            }

            return stats;
        }

        private bool TryCreateNonNegativeAdditiveStepEvaluator(
            ArithmeticExpression baseExpression,
            ArithmeticExpression recursiveExpression,
            IReadOnlyDictionary<object, PathAwareSuccessorBucket> succIndex,
            IReadOnlyDictionary<object, List<object[]>> auxIndex,
            bool positiveStepProven,
            out PositiveStepEvaluator? evaluator,
            out AdditiveMinStepSafety stepSafety)
        {
            evaluator = null;
            stepSafety = AdditiveMinStepSafety.StrictlyPositive;
            if (!TryExtractMinStepExpression(baseExpression, recursiveExpression, out var stepExpression))
            {
                return false;
            }

            if (positiveStepProven)
            {
                evaluator = (current, next, auxValue) =>
                {
                    var evalTuple = new object[] { current, next, 0, auxValue };
                    var stepObject = EvaluateArithmeticExpression(stepExpression, evalTuple);
                    return Convert.ToDouble(stepObject, CultureInfo.InvariantCulture);
                };
                return true;
            }

            var strictlyPositive = true;
            foreach (var (currentKey, edgeBucket) in succIndex)
            {
                if (!auxIndex.TryGetValue(currentKey, out var auxBucket))
                {
                    continue;
                }

                foreach (var next in edgeBucket.Targets)
                {
                    foreach (var auxRow in auxBucket)
                    {
                        if (auxRow is null || auxRow.Length < 2)
                        {
                            continue;
                        }

                        var evalTuple = new object[] { edgeBucket.Source!, next!, 0, auxRow[1]! };
                        var stepObject = EvaluateArithmeticExpression(stepExpression, evalTuple);
                        double step;
                        try
                        {
                            step = Convert.ToDouble(stepObject, CultureInfo.InvariantCulture);
                        }
                        catch
                        {
                            return false;
                        }

                        if (!(step >= 0d))
                        {
                            return false;
                        }

                        if (!(step > 0d))
                        {
                            strictlyPositive = false;
                        }
                    }
                }
            }

            stepSafety = strictlyPositive
                ? AdditiveMinStepSafety.StrictlyPositive
                : AdditiveMinStepSafety.NonNegative;
            evaluator = (current, next, auxValue) =>
            {
                var evalTuple = new object[] { current, next, 0, auxValue };
                var stepObject = EvaluateArithmeticExpression(stepExpression, evalTuple);
                return Convert.ToDouble(stepObject, CultureInfo.InvariantCulture);
            };
            return true;
        }

        private static bool TryExtractMinStepExpression(
            ArithmeticExpression baseExpression,
            ArithmeticExpression recursiveExpression,
            out ArithmeticExpression stepExpression)
        {
            stepExpression = baseExpression;
            if (ReferencesAccumulator(baseExpression))
            {
                return false;
            }

            if (recursiveExpression is not BinaryArithmeticExpression { Operator: ArithmeticBinaryOperator.Add } add)
            {
                return false;
            }

            if (add.Left is ColumnExpression { Index: 2 } && ArithmeticExpressionStructuralEquals(baseExpression, add.Right))
            {
                stepExpression = add.Right;
                return true;
            }

            if (add.Right is ColumnExpression { Index: 2 } && ArithmeticExpressionStructuralEquals(baseExpression, add.Left))
            {
                stepExpression = add.Left;
                return true;
            }

            return false;
        }

        private static bool ReferencesAccumulator(ArithmeticExpression expression) =>
            expression switch
            {
                ColumnExpression { Index: 2 } => true,
                ColumnExpression => false,
                ConstantExpression => false,
                UnaryArithmeticExpression unary => ReferencesAccumulator(unary.Operand),
                BinaryArithmeticExpression binary => ReferencesAccumulator(binary.Left) || ReferencesAccumulator(binary.Right),
                _ => true
            };

        private static bool ArithmeticExpressionStructuralEquals(ArithmeticExpression left, ArithmeticExpression right) =>
            (left, right) switch
            {
                (ColumnExpression l, ColumnExpression r) => l.Index == r.Index,
                (ConstantExpression l, ConstantExpression r) => Equals(l.Value, r.Value),
                (UnaryArithmeticExpression l, UnaryArithmeticExpression r) =>
                    l.Operator == r.Operator && ArithmeticExpressionStructuralEquals(l.Operand, r.Operand),
                (BinaryArithmeticExpression l, BinaryArithmeticExpression r) =>
                    l.Operator == r.Operator &&
                    ArithmeticExpressionStructuralEquals(l.Left, r.Left) &&
                    ArithmeticExpressionStructuralEquals(l.Right, r.Right),
                _ => false
            };

        private void AppendPathAwareAccumulationRowsForSeed(
            object? seed,
            IReadOnlyDictionary<object, PathAwareSuccessorBucket> succIndex,
            IReadOnlyDictionary<object, List<object[]>> auxIndex,
            ArithmeticExpression baseExpression,
            ArithmeticExpression recursiveExpression,
            ICollection<object[]> output,
            TableMode accumulatorMode,
            int maxDepth = 0,
            PathAwareMinFrontierMetrics? frontierMetrics = null)
        {
            var useMinPruning = accumulatorMode == TableMode.Min;
            var preserveAllPaths = !useMinPruning;
            var bestKnown = useMinPruning
                ? new Dictionary<object?, object>()
                : null;
            var frontier = useMinPruning
                ? new Dictionary<object?, PathAwareMinFrontierBucket>()
                : null;
            var nodeIds = new Dictionary<object, int>();
            var nextNodeId = 0;

            int GetNodeId(object? value)
            {
                var key = value ?? NullFactIndexKey;
                if (nodeIds.TryGetValue(key, out var existing))
                {
                    return existing;
                }

                var id = nextNodeId++;
                nodeIds[key] = id;
                return id;
            }

            var initialPath = CompactVisitedPath.Create(GetNodeId(seed));
            var initialState = new VisitedAccumulatorState(0, initialPath);
            var stack = new Stack<(object? Node, VisitedAccumulatorState State, int Depth)>();
            stack.Push((seed, initialState, 0));

            while (stack.Count > 0)
            {
                var (current, state, depth) = stack.Pop();
                if (useMinPruning && current is not null && IsDominatedMinState(frontier!, current, state, frontierMetrics))
                {
                    frontierMetrics?.RecordDominatedState();
                    continue;
                }

                var accumulator = state.Accumulator;
                var visited = state.Path;

                var currentKey = current ?? NullFactIndexKey;
                if (!succIndex.TryGetValue(currentKey, out var edgeBucket) || !auxIndex.TryGetValue(currentKey, out var auxBucket))
                {
                    continue;
                }

                for (var edgeIndex = edgeBucket.Targets.Count - 1; edgeIndex >= 0; edgeIndex--)
                {
                    var next = edgeBucket.Targets[edgeIndex];
                    var nextId = GetNodeId(next);
                    if (visited.Contains(nextId))
                    {
                        continue;
                    }

                    var nextDepth = depth + 1;
                    if (maxDepth > 0 && nextDepth > maxDepth)
                    {
                        continue;
                    }

                    for (var auxIndexPos = auxBucket.Count - 1; auxIndexPos >= 0; auxIndexPos--)
                    {
                        var auxRow = auxBucket[auxIndexPos];
                        if (auxRow is null || auxRow.Length < 2)
                        {
                            continue;
                        }

                        var auxValue = auxRow[1];
                        var evalTuple = new object[] { current!, next!, accumulator, auxValue! };
                        var nextAccumulator = depth == 0
                            ? EvaluateArithmeticExpression(baseExpression, evalTuple)
                            : EvaluateArithmeticExpression(recursiveExpression, evalTuple);
                        var nextState = new VisitedAccumulatorState(nextAccumulator, visited.Extend(nextId));

                        if (preserveAllPaths)
                        {
                            output.Add(new object[] { seed!, next!, nextAccumulator });
                        }
                        else
                        {
                            frontierMetrics?.RecordCandidate();
                            if (IsDominatedMinState(frontier!, next, nextState, frontierMetrics))
                            {
                                frontierMetrics?.RecordDominatedState();
                                continue;
                            }

                            RecordMinState(frontier!, next, nextState, frontierMetrics);
                            if (!bestKnown!.TryGetValue(next, out var bestAccumulator) ||
                                CompareValues(nextAccumulator, bestAccumulator) < 0)
                            {
                                bestKnown[next] = nextAccumulator;
                            }
                        }

                        stack.Push((next, nextState, nextDepth));
                    }
                }
            }

            if (!preserveAllPaths && bestKnown is not null)
            {
                frontierMetrics?.RecordBucketSnapshot(frontier!);
                var bestRows = bestKnown
                    .OrderBy(kvp => kvp.Key, Comparer<object?>.Create(CompareCacheSeedValues))
                    .ToList();

                foreach (var (target, accumulator) in bestRows)
                {
                    output.Add(new object[] { seed!, target!, accumulator });
                }
            }
        }

        private static bool IsDominatedMinState(
            IReadOnlyDictionary<object?, PathAwareMinFrontierBucket> frontier,
            object? node,
            VisitedAccumulatorState state,
            PathAwareMinFrontierMetrics? metrics = null)
        {
            metrics?.RecordDominanceCheck();
            return frontier.TryGetValue(node, out var bucket) &&
                bucket.IsDominatedByExisting(state, metrics);
        }

        private static void RecordMinState(
            IDictionary<object?, PathAwareMinFrontierBucket> frontier,
            object? node,
            VisitedAccumulatorState state,
            PathAwareMinFrontierMetrics? metrics = null)
        {
            if (!frontier.TryGetValue(node, out var bucket))
            {
                bucket = new PathAwareMinFrontierBucket();
                frontier[node] = bucket;
            }

            bucket.AddStateAndRemoveDominated(state, metrics);
            metrics?.RecordRecordedState();
        }

        private sealed class PathAwareMinFrontierBucket
        {
            private readonly Dictionary<int, PathAwareMinFrontierCountBucket> _statesByCount = new();

            public bool IsDominatedByExisting(VisitedAccumulatorState state, PathAwareMinFrontierMetrics? metrics)
            {
                for (var pathCount = 1; pathCount <= state.Path.Count; pathCount++)
                {
                    if (!_statesByCount.TryGetValue(pathCount, out var countBucket))
                    {
                        continue;
                    }

                    if (pathCount == state.Path.Count)
                    {
                        return countBucket.IsDominatedBySameFingerprint(state, metrics);
                    }

                    if (countBucket.IsDominatedByAny(state, metrics))
                    {
                        return true;
                    }
                }

                return false;
            }

            public void AddStateAndRemoveDominated(VisitedAccumulatorState state, PathAwareMinFrontierMetrics? metrics)
            {
                var pathCounts = _statesByCount.Keys.ToList();
                foreach (var pathCount in pathCounts)
                {
                    if (pathCount < state.Path.Count ||
                        !_statesByCount.TryGetValue(pathCount, out var countBucket))
                    {
                        continue;
                    }

                    if (pathCount == state.Path.Count)
                    {
                        countBucket.RemoveDominatedBySameFingerprint(state, metrics);
                    }
                    else
                    {
                        countBucket.RemoveDominatedByAny(state, metrics);
                    }

                    if (countBucket.IsEmpty)
                    {
                        _statesByCount.Remove(pathCount);
                    }
                }

                if (!_statesByCount.TryGetValue(state.Path.Count, out var bucketByCount))
                {
                    bucketByCount = new PathAwareMinFrontierCountBucket();
                    _statesByCount[state.Path.Count] = bucketByCount;
                }

                bucketByCount.Add(state);
            }

            public void RecordSnapshot(PathAwareMinFrontierMetrics metrics)
            {
                foreach (var countBucket in _statesByCount.Values)
                {
                    countBucket.RecordSnapshot(metrics);
                }
            }
        }

        private sealed class PathAwareMinFrontierCountBucket
        {
            private readonly List<VisitedAccumulatorState> _states = new();
            private readonly Dictionary<ulong, int> _fingerprintCounts = new();

            public bool IsEmpty => _states.Count == 0;

            public void Add(VisitedAccumulatorState state)
            {
                _states.Add(state);
                var fingerprint = state.Path.Fingerprint;
                _fingerprintCounts.TryGetValue(fingerprint, out var count);
                _fingerprintCounts[fingerprint] = count + 1;
            }

            public bool IsDominatedBySameFingerprint(VisitedAccumulatorState state, PathAwareMinFrontierMetrics? metrics)
            {
                return _fingerprintCounts.ContainsKey(state.Path.Fingerprint) &&
                    IsDominatedByAny(state, metrics, requireSameFingerprint: true);
            }

            public bool IsDominatedByAny(VisitedAccumulatorState state, PathAwareMinFrontierMetrics? metrics)
            {
                return IsDominatedByAny(state, metrics, requireSameFingerprint: false);
            }

            public void RemoveDominatedBySameFingerprint(VisitedAccumulatorState state, PathAwareMinFrontierMetrics? metrics)
            {
                if (_fingerprintCounts.ContainsKey(state.Path.Fingerprint))
                {
                    RemoveDominated(state, metrics, requireSameFingerprint: true);
                }
            }

            public void RemoveDominatedByAny(VisitedAccumulatorState state, PathAwareMinFrontierMetrics? metrics)
            {
                RemoveDominated(state, metrics, requireSameFingerprint: false);
            }

            public void RecordSnapshot(PathAwareMinFrontierMetrics metrics)
            {
                foreach (var count in _fingerprintCounts.Values)
                {
                    metrics.RecordPartitionBucket(count);
                }
            }

            private bool IsDominatedByAny(
                VisitedAccumulatorState state,
                PathAwareMinFrontierMetrics? metrics,
                bool requireSameFingerprint)
            {
                foreach (var candidate in _states)
                {
                    if (requireSameFingerprint && candidate.Path.Fingerprint != state.Path.Fingerprint)
                    {
                        continue;
                    }

                    metrics?.RecordDominanceCandidateCheck();
                    var sameState = ReferenceEquals(candidate.Path, state.Path) && Equals(candidate.Accumulator, state.Accumulator);
                    if (sameState)
                    {
                        continue;
                    }

                    if ((candidate.Path.MaskA & ~state.Path.MaskA) != 0 || (candidate.Path.MaskB & ~state.Path.MaskB) != 0)
                    {
                        continue;
                    }

                    if (CompareValues(candidate.Accumulator, state.Accumulator) <= 0 &&
                        (metrics?.RecordSubsetCheckAndReturnTrue() ?? true) &&
                        candidate.Path.IsSubsetOf(state.Path))
                    {
                        return true;
                    }
                }

                return false;
            }

            private void RemoveDominated(
                VisitedAccumulatorState state,
                PathAwareMinFrontierMetrics? metrics,
                bool requireSameFingerprint)
            {
                for (var i = _states.Count - 1; i >= 0; i--)
                {
                    var existing = _states[i];
                    if (requireSameFingerprint && existing.Path.Fingerprint != state.Path.Fingerprint)
                    {
                        continue;
                    }

                    if ((state.Path.MaskA & ~existing.Path.MaskA) != 0 || (state.Path.MaskB & ~existing.Path.MaskB) != 0)
                    {
                        continue;
                    }

                    if (CompareValues(state.Accumulator, existing.Accumulator) <= 0 &&
                        (metrics?.RecordSubsetCheckAndReturnTrue() ?? true) &&
                        state.Path.IsSubsetOf(existing.Path))
                    {
                        RemoveAt(i);
                        metrics?.RecordRemovedState();
                    }
                }
            }

            private void RemoveAt(int index)
            {
                var fingerprint = _states[index].Path.Fingerprint;
                _states.RemoveAt(index);

                var count = _fingerprintCounts[fingerprint];
                if (count <= 1)
                {
                    _fingerprintCounts.Remove(fingerprint);
                }
                else
                {
                    _fingerprintCounts[fingerprint] = count - 1;
                }
            }
        }

        private sealed class PathAwareMinFrontierMetrics
        {
            public long CandidateCount { get; private set; }

            public long DominanceCheckCount { get; private set; }

            public long DominanceCandidateCheckCount { get; private set; }

            public long SubsetCheckCount { get; private set; }

            public long DominatedStateCount { get; private set; }

            public long RecordedStateCount { get; private set; }

            public long RemovedStateCount { get; private set; }

            public long TargetBucketCount { get; private set; }

            public long BucketCount { get; private set; }

            public long BucketStateCount { get; private set; }

            public long MaxBucketSize { get; private set; }

            public void RecordCandidate() => CandidateCount++;

            public void RecordDominanceCheck() => DominanceCheckCount++;

            public void RecordDominanceCandidateCheck() => DominanceCandidateCheckCount++;

            public bool RecordSubsetCheckAndReturnTrue()
            {
                SubsetCheckCount++;
                return true;
            }

            public void RecordDominatedState() => DominatedStateCount++;

            public void RecordRecordedState() => RecordedStateCount++;

            public void RecordRemovedState() => RemovedStateCount++;

            public void RecordBucketSnapshot(IReadOnlyDictionary<object?, PathAwareMinFrontierBucket> frontier)
            {
                TargetBucketCount += frontier.Count;
                foreach (var bucket in frontier.Values)
                {
                    bucket.RecordSnapshot(this);
                }
            }

            public void RecordPartitionBucket(int count)
            {
                BucketCount++;
                BucketStateCount += count;
                if (count > MaxBucketSize)
                {
                    MaxBucketSize = count;
                }
            }
        }

        private sealed record VisitedAccumulatorState(
            object Accumulator,
            CompactVisitedPath Path);

        private sealed class CompactVisitedPath
        {
            private readonly int[] _nodeIds;

            private CompactVisitedPath(int[] nodeIds, int count, ulong maskA, ulong maskB, ulong fingerprint)
            {
                _nodeIds = nodeIds;
                Count = count;
                MaskA = maskA;
                MaskB = maskB;
                Fingerprint = fingerprint;
            }

            public int Count { get; }

            public ulong MaskA { get; }

            public ulong MaskB { get; }

            public ulong Fingerprint { get; }

            public static CompactVisitedPath Create(int nodeId)
            {
                return new CompactVisitedPath(
                    new[] { nodeId },
                    1,
                    ComputeVisitedMaskA(nodeId),
                    ComputeVisitedMaskB(nodeId),
                    ComputeVisitedFingerprint(nodeId));
            }

            public bool Contains(int nodeId)
            {
                var maskA = ComputeVisitedMaskA(nodeId);
                if ((MaskA & maskA) == 0)
                {
                    return false;
                }

                var maskB = ComputeVisitedMaskB(nodeId);
                if ((MaskB & maskB) == 0)
                {
                    return false;
                }

                for (var i = 0; i < Count; i++)
                {
                    if (_nodeIds[i] == nodeId)
                    {
                        return true;
                    }
                }

                return false;
            }

            public CompactVisitedPath Extend(int nodeId)
            {
                var nodeIds = new int[Count + 1];
                Array.Copy(_nodeIds, nodeIds, Count);
                nodeIds[Count] = nodeId;
                return new CompactVisitedPath(
                    nodeIds,
                    Count + 1,
                    MaskA | ComputeVisitedMaskA(nodeId),
                    MaskB | ComputeVisitedMaskB(nodeId),
                    Fingerprint ^ ComputeVisitedFingerprint(nodeId));
            }

            public bool IsSubsetOf(CompactVisitedPath other)
            {
                if (Count > other.Count)
                {
                    return false;
                }

                if ((MaskA & ~other.MaskA) != 0 || (MaskB & ~other.MaskB) != 0)
                {
                    return false;
                }

                for (var i = 0; i < Count; i++)
                {
                    var found = false;
                    var value = _nodeIds[i];
                    for (var j = 0; j < other.Count; j++)
                    {
                        if (value == other._nodeIds[j])
                        {
                            found = true;
                            break;
                        }
                    }

                    if (!found)
                    {
                        return false;
                    }
                }

                return true;
            }
        }

        private static ulong ComputeVisitedMaskA(int value)
        {
            return 1UL << (value & 63);
        }

        private static ulong ComputeVisitedMaskB(int value)
        {
            return 1UL << ((value >> 6) & 63);
        }

        private static ulong ComputeVisitedFingerprint(int value)
        {
            var z = unchecked((ulong)(uint)value + 0x9E3779B97F4A7C15UL);
            z = unchecked((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL);
            z = unchecked((z ^ (z >> 27)) * 0x94D049BB133111EBUL);
            return z ^ (z >> 31);
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

                var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);

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

        private IEnumerable<object[]> ExecuteSeedGroupedTransitiveClosure(SeedGroupedTransitiveClosureNode closure, EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));

            var width = closure.Predicate.Arity;
            if (width < 2)
            {
                return Array.Empty<object[]>();
            }

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                trace?.RecordStrategy(closure, "SeedGroupedTransitiveClosure");

                var predicate = closure.Predicate;
                var traceKey = $"{predicate.Name}/{predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:seeds={closure.SeedRelation.Name}/{closure.SeedRelation.Arity}";
                var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);
                var seeds = GetClosureFactsList(closure.SeedRelation, ClosureRelationAccessKind.Support, context, closure);

                const int MaxDagNodeCount = 16384;
                const int MaxDagGroupCount = 4096;
                const int MinDagGroupCount = 64;
                const int MinDagEdgeCount = 8192;
                if (seeds.Count >= MinDagGroupCount &&
                    edges.Count >= MinDagEdgeCount &&
                    TryBuildProjectGroupedDagReachRows(edges, seeds, MaxDagNodeCount, MaxDagGroupCount, out var dagRows))
                {
                    trace?.RecordCacheLookup("SeedGroupedTransitiveClosure", traceKey, hit: false, built: true);
                    trace?.RecordStrategy(closure, "SeedGroupedTransitiveClosureDag");
                    return dagRows;
                }

                trace?.RecordCacheLookup("SeedGroupedTransitiveClosure", traceKey, hit: false, built: true);

                var succIndex = GetFactIndex(closure.EdgeRelation, 0, edges, context);
                var wrapperComparer = new RowWrapperComparer(StructuralArrayComparer.Instance);
                var visited = new HashSet<RowWrapper>(wrapperComparer);
                var totalRows = new List<object[]>();
                var delta = new List<object[]>();

                foreach (var seed in seeds)
                {
                    if (seed is null || seed.Length < 2)
                    {
                        continue;
                    }

                    var group = seed[0];
                    var node = seed[1];
                    var key = new object[] { group!, node! };
                    if (visited.Add(new RowWrapper(key)))
                    {
                        totalRows.Add((object[])key.Clone());
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
                        var node = pair[1];
                        var lookupKey = node ?? NullFactIndexKey;
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
                            var nextKey = new object[] { pair[0], next! };
                            if (visited.Add(new RowWrapper(nextKey)))
                            {
                                totalRows.Add((object[])nextKey.Clone());
                                nextDelta.Add(nextKey);
                            }
                        }
                    }

                    delta = nextDelta;
                    trace?.RecordFixpointIteration(closure, predicate, iteration, delta.Count, totalRows.Count);
                }

                return totalRows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private IEnumerable<object[]> ExecuteSeedGroupedTransitiveClosureCount(SeedGroupedTransitiveClosureCountNode closure, EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));

            var width = closure.Predicate.Arity;
            if (width < 2)
            {
                return Array.Empty<object[]>();
            }

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                trace?.RecordStrategy(closure, "SeedGroupedTransitiveClosureCount");

                var predicate = closure.Predicate;
                var cacheKey = (closure.EdgeRelation, closure.SeedRelation, predicate);
                var traceKey = $"{predicate.Name}/{predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:seeds={closure.SeedRelation.Name}/{closure.SeedRelation.Arity}";
                if (context.SeedGroupedTransitiveClosureCountResults.TryGetValue(cacheKey, out var cachedRows))
                {
                    trace?.RecordCacheLookup("SeedGroupedTransitiveClosureCount", traceKey, hit: true, built: false);
                    return cachedRows;
                }

                trace?.RecordCacheLookup("SeedGroupedTransitiveClosureCount", traceKey, hit: false, built: true);

                const int MaxDagNodeCount = 16384;
                const int MaxDagGroupCount = 4096;
                const int MinDagGroupCount = 64;
                const int MinDagEdgeCount = 8192;
                const int DagProbeEdgeRowLimit = 256;
                const int DagProbeSeedRowLimit = 128;

                var hasStreamingEdge = TryGetDelimitedSource(closure.EdgeRelation, RelationRetentionMode.Streaming, out var edgeSource);
                var hasStreamingSeed = TryGetDelimitedSource(closure.SeedRelation, RelationRetentionMode.Streaming, out var seedSource);
                var hasStreaming = hasStreamingEdge && hasStreamingSeed;
                var hasReplayableEdge = TryGetReplayableSource(closure.EdgeRelation, context, out var edgeReplayableSource);
                var hasReplayableSeed = TryGetReplayableSource(closure.SeedRelation, context, out var seedReplayableSource);
                var hasReplayable = hasReplayableEdge && hasReplayableSeed;
                var hasExternalEdge = TryGetExternalFacts(closure.EdgeRelation, out var externalEdgeFacts);
                var hasExternalSeed = TryGetExternalFacts(closure.SeedRelation, out var externalSeedFacts);
                var hasExternal = hasExternalEdge && hasExternalSeed;
                var edgeConcreteReplayable = edgeReplayableSource as ReplayableRelationSource;
                var seedConcreteReplayable = seedReplayableSource as ReplayableRelationSource;
                var replayableMaterialized = edgeConcreteReplayable?.IsMaterialized == true && seedConcreteReplayable?.IsMaterialized == true;
                long? relationBytes = hasStreaming &&
                    !string.IsNullOrEmpty(edgeSource.InputPath) && File.Exists(edgeSource.InputPath) &&
                    !string.IsNullOrEmpty(seedSource.InputPath) && File.Exists(seedSource.InputPath)
                    ? new FileInfo(edgeSource.InputPath).Length + new FileInfo(seedSource.InputPath).Length
                    : (long?)null;

                Func<TimeSpan>? measureStreamingProbe = hasStreaming
                    ? () => MeasureElapsed(() => ProbeDagDelimited(edgeSource, seedSource, DagProbeEdgeRowLimit, DagProbeSeedRowLimit))
                    : null;
                Func<TimeSpan>? measureReplayableProbe = hasReplayable
                    ? () => MeasureElapsed(() =>
                    {
                        var edgeProbeRows = edgeConcreteReplayable is not null
                            ? edgeConcreteReplayable.ProbeMaterialize(DagProbeEdgeRowLimit)
                            : edgeReplayableSource.Stream().Take(DagProbeEdgeRowLimit).ToList();
                        var seedProbeRows = seedConcreteReplayable is not null
                            ? seedConcreteReplayable.ProbeMaterialize(DagProbeSeedRowLimit)
                            : seedReplayableSource.Stream().Take(DagProbeSeedRowLimit).ToList();
                        ProbeDagRows(edgeProbeRows, seedProbeRows, DagProbeEdgeRowLimit, DagProbeSeedRowLimit);
                    })
                    : null;
                Func<TimeSpan>? measureExternalProbe = hasExternal
                    ? () => MeasureElapsed(() => ProbeDagRows(externalEdgeFacts, externalSeedFacts, DagProbeEdgeRowLimit, DagProbeSeedRowLimit))
                    : null;

                var selection = ResolveDagRelationRetentionStrategy(
                    trace,
                    closure,
                    _dagRelationRetentionStrategy,
                    relationBytes,
                    hasStreaming,
                    hasReplayable,
                    hasExternal,
                    replayableMaterialized,
                    measureStreamingProbe,
                    measureReplayableProbe,
                    measureExternalProbe);
                RecordDagRelationRetentionStrategy(trace, closure, selection);
                var materializationPlan = ResolveMaterializationPlan(
                    trace,
                    closure,
                    ToRelationRetentionSelection(selection));
                RecordMaterializationPlanStrategy(trace, closure, "SeedGroupedTransitiveClosureCount", materializationPlan);

                List<object[]>? edges = null;
                List<object[]>? seeds = null;

                if (selection.Strategy == DagRelationRetentionStrategy.StreamingDirect && hasStreaming)
                {
                    List<object[]> streamedDagRows = new();
                    if (MeasurePhase(trace, closure, "dag_build_streaming_direct", () =>
                            TryBuildProjectGroupedDagReachCountRowsFromDelimitedSources(edgeSource, seedSource, MinDagEdgeCount, MinDagGroupCount, MaxDagNodeCount, MaxDagGroupCount, out streamedDagRows)))
                    {
                        trace?.RecordStrategy(closure, "SeedGroupedTransitiveClosureCountDagStreamed");
                        context.SeedGroupedTransitiveClosureCountResults[cacheKey] = streamedDagRows;
                        return streamedDagRows;
                    }
                }

                if (selection.Strategy == DagRelationRetentionStrategy.ReplayableBuffer && hasReplayable)
                {
                    edges = MeasurePhase(trace, closure, "dag_materialize_replayable_edges", () => edgeReplayableSource.Materialize());
                    seeds = MeasurePhase(trace, closure, "dag_materialize_replayable_seeds", () => seedReplayableSource.Materialize());
                }
                else if (selection.Strategy == DagRelationRetentionStrategy.ExternalMaterialized && hasExternal)
                {
                    edges = externalEdgeFacts as List<object[]> ?? MeasurePhase(trace, closure, "dag_materialize_external_edges", () => externalEdgeFacts.ToList());
                    seeds = externalSeedFacts as List<object[]> ?? MeasurePhase(trace, closure, "dag_materialize_external_seeds", () => externalSeedFacts.ToList());
                }

                edges ??= GetDagFactsList(closure.EdgeRelation, selection.Strategy, context, closure, edgeRelation: true);
                seeds ??= GetDagFactsList(closure.SeedRelation, selection.Strategy, context, closure, edgeRelation: false);
                if (seeds.Count >= MinDagGroupCount &&
                    edges.Count >= MinDagEdgeCount)
                {
                    List<object[]> dagRows = new();
                    var buildPhase = selection.Strategy switch
                    {
                        DagRelationRetentionStrategy.StreamingDirect when hasStreaming => "dag_build_streaming_fallback",
                        DagRelationRetentionStrategy.ReplayableBuffer when hasReplayable => "dag_build_replayable_buffer",
                        DagRelationRetentionStrategy.ExternalMaterialized when hasExternal => "dag_build_external_materialized",
                        _ => "dag_build_materialized_fallback"
                    };
                    if (MeasurePhase(trace, closure, buildPhase, () =>
                            TryBuildProjectGroupedDagReachCountRows(edges, seeds, MaxDagNodeCount, MaxDagGroupCount, out dagRows)))
                    {
                        trace?.RecordStrategy(closure, "SeedGroupedTransitiveClosureCountDag");
                        context.SeedGroupedTransitiveClosureCountResults[cacheKey] = dagRows;
                        return dagRows;
                    }
                }

                var succIndex = GetFactIndex(closure.EdgeRelation, 0, edges, context);
                var wrapperComparer = new RowWrapperComparer(StructuralArrayComparer.Instance);
                var visited = new HashSet<RowWrapper>(wrapperComparer);
                var groupCounts = new Dictionary<object?, int>();
                var delta = new List<object[]>();

                foreach (var seed in seeds)
                {
                    if (seed is null || seed.Length < 2)
                    {
                        continue;
                    }

                    var group = seed[0];
                    var node = seed[1];
                    var key = new object[] { group!, node! };
                    if (visited.Add(new RowWrapper(key)))
                    {
                        delta.Add(key);
                        groupCounts[group] = groupCounts.TryGetValue(group, out var count) ? count + 1 : 1;
                    }
                }

                var iteration = 0;
                trace?.RecordFixpointIteration(closure, predicate, iteration, delta.Count, visited.Count);

                while (delta.Count > 0)
                {
                    iteration++;
                    var nextDelta = new List<object[]>();

                    foreach (var pair in delta)
                    {
                        var node = pair[1];
                        var lookupKey = node ?? NullFactIndexKey;
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
                            var nextKey = new object[] { pair[0], next! };
                            if (visited.Add(new RowWrapper(nextKey)))
                            {
                                nextDelta.Add(nextKey);
                                var group = pair[0];
                                groupCounts[group] = groupCounts.TryGetValue(group, out var count) ? count + 1 : 1;
                            }
                        }
                    }

                    delta = nextDelta;
                    trace?.RecordFixpointIteration(closure, predicate, iteration, delta.Count, visited.Count);
                }

                var rows = groupCounts
                    .Select(kvp => new object[] { kvp.Key!, kvp.Value })
                    .ToList();
                context.SeedGroupedTransitiveClosureCountResults[cacheKey] = rows;
                return rows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private IEnumerable<object[]> ExecuteSeedGroupedDagLongestDepth(SeedGroupedDagLongestDepthNode closure, EvaluationContext? parentContext)
        {
            if (closure is null) throw new ArgumentNullException(nameof(closure));

            var width = closure.Predicate.Arity;
            if (width < 2)
            {
                return Array.Empty<object[]>();
            }

            var context = parentContext ?? new EvaluationContext();
            context.FixpointDepth++;
            try
            {
                var trace = context.Trace;
                trace?.RecordStrategy(closure, "SeedGroupedDagLongestDepth");

                var predicate = closure.Predicate;
                var cacheKey = (closure.EdgeRelation, closure.SeedRelation, predicate);
                var traceKey = $"{predicate.Name}/{predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:seeds={closure.SeedRelation.Name}/{closure.SeedRelation.Arity}";
                if (context.SeedGroupedDagLongestDepthResults.TryGetValue(cacheKey, out var cachedRows))
                {
                    trace?.RecordCacheLookup("SeedGroupedDagLongestDepth", traceKey, hit: true, built: false);
                    return cachedRows;
                }

                trace?.RecordCacheLookup("SeedGroupedDagLongestDepth", traceKey, hit: false, built: true);

                const int MaxDagNodeCount = 65536;
                const int DagProbeEdgeRowLimit = 256;
                const int DagProbeSeedRowLimit = 128;

                var hasStreamingEdge = TryGetDelimitedSource(closure.EdgeRelation, RelationRetentionMode.Streaming, out var edgeSource);
                var hasStreamingSeed = TryGetDelimitedSource(closure.SeedRelation, RelationRetentionMode.Streaming, out var seedSource);
                var hasStreaming = hasStreamingEdge && hasStreamingSeed;
                var hasReplayableEdge = TryGetReplayableSource(closure.EdgeRelation, context, out var edgeReplayableSource);
                var hasReplayableSeed = TryGetReplayableSource(closure.SeedRelation, context, out var seedReplayableSource);
                var hasReplayable = hasReplayableEdge && hasReplayableSeed;
                var hasExternalEdge = TryGetExternalFacts(closure.EdgeRelation, out var externalEdgeFacts);
                var hasExternalSeed = TryGetExternalFacts(closure.SeedRelation, out var externalSeedFacts);
                var hasExternal = hasExternalEdge && hasExternalSeed;
                var edgeConcreteReplayable = edgeReplayableSource as ReplayableRelationSource;
                var seedConcreteReplayable = seedReplayableSource as ReplayableRelationSource;
                var replayableMaterialized = edgeConcreteReplayable?.IsMaterialized == true && seedConcreteReplayable?.IsMaterialized == true;
                long? relationBytes = hasStreaming &&
                    !string.IsNullOrEmpty(edgeSource.InputPath) && File.Exists(edgeSource.InputPath) &&
                    !string.IsNullOrEmpty(seedSource.InputPath) && File.Exists(seedSource.InputPath)
                    ? new FileInfo(edgeSource.InputPath).Length + new FileInfo(seedSource.InputPath).Length
                    : (long?)null;

                Func<TimeSpan>? measureStreamingProbe = hasStreaming
                    ? () => MeasureElapsed(() => ProbeDagDelimited(edgeSource, seedSource, DagProbeEdgeRowLimit, DagProbeSeedRowLimit))
                    : null;
                Func<TimeSpan>? measureReplayableProbe = hasReplayable
                    ? () => MeasureElapsed(() =>
                    {
                        var edgeProbeRows = edgeConcreteReplayable is not null
                            ? edgeConcreteReplayable.ProbeMaterialize(DagProbeEdgeRowLimit)
                            : edgeReplayableSource.Stream().Take(DagProbeEdgeRowLimit).ToList();
                        var seedProbeRows = seedConcreteReplayable is not null
                            ? seedConcreteReplayable.ProbeMaterialize(DagProbeSeedRowLimit)
                            : seedReplayableSource.Stream().Take(DagProbeSeedRowLimit).ToList();
                        ProbeDagRows(edgeProbeRows, seedProbeRows, DagProbeEdgeRowLimit, DagProbeSeedRowLimit);
                    })
                    : null;
                Func<TimeSpan>? measureExternalProbe = hasExternal
                    ? () => MeasureElapsed(() => ProbeDagRows(externalEdgeFacts, externalSeedFacts, DagProbeEdgeRowLimit, DagProbeSeedRowLimit))
                    : null;

                var selection = ResolveDagRelationRetentionStrategy(
                    trace,
                    closure,
                    _dagRelationRetentionStrategy,
                    relationBytes,
                    hasStreaming,
                    hasReplayable,
                    hasExternal,
                    replayableMaterialized,
                    measureStreamingProbe,
                    measureReplayableProbe,
                    measureExternalProbe);
                RecordDagRelationRetentionStrategy(trace, closure, selection);
                var materializationPlan = ResolveMaterializationPlan(
                    trace,
                    closure,
                    ToRelationRetentionSelection(selection));
                RecordMaterializationPlanStrategy(trace, closure, "SeedGroupedDagLongestDepth", materializationPlan);

                if (selection.Strategy == DagRelationRetentionStrategy.StreamingDirect && hasStreaming)
                {
                    List<object[]> streamedRows = new();
                    if (MeasurePhase(trace, closure, "dag_build_streaming_direct", () =>
                            TryBuildSeedGroupedDagLongestDepthRowsFromDelimitedSources(closure, trace, edgeSource, seedSource, MaxDagNodeCount, out streamedRows)))
                    {
                        context.SeedGroupedDagLongestDepthResults[cacheKey] = streamedRows;
                        return streamedRows;
                    }
                }

                List<object[]>? edges = null;
                List<object[]>? seeds = null;
                string buildPhase = "dag_build_materialized_fallback";
                if (selection.Strategy == DagRelationRetentionStrategy.ReplayableBuffer && hasReplayable)
                {
                    edges = MeasurePhase(trace, closure, "dag_materialize_replayable_edges", () => edgeReplayableSource.Materialize());
                    seeds = MeasurePhase(trace, closure, "dag_materialize_replayable_seeds", () => seedReplayableSource.Materialize());
                    buildPhase = "dag_build_replayable_buffer";
                }
                else if (selection.Strategy == DagRelationRetentionStrategy.ExternalMaterialized && hasExternal)
                {
                    edges = externalEdgeFacts as List<object[]> ?? MeasurePhase(trace, closure, "dag_materialize_external_edges", () => externalEdgeFacts.ToList());
                    seeds = externalSeedFacts as List<object[]> ?? MeasurePhase(trace, closure, "dag_materialize_external_seeds", () => externalSeedFacts.ToList());
                    buildPhase = "dag_build_external_materialized";
                }

                edges ??= GetDagFactsList(closure.EdgeRelation, selection.Strategy, context, closure, edgeRelation: true);
                seeds ??= GetDagFactsList(closure.SeedRelation, selection.Strategy, context, closure, edgeRelation: false);
                buildPhase = selection.Strategy switch
                {
                    DagRelationRetentionStrategy.StreamingDirect when hasStreaming => "dag_build_streaming_fallback",
                    DagRelationRetentionStrategy.ReplayableBuffer when hasReplayable => "dag_build_replayable_buffer",
                    DagRelationRetentionStrategy.ExternalMaterialized when hasExternal => "dag_build_external_materialized",
                    _ => buildPhase
                };
                List<object[]> rows = new();
                if (!MeasurePhase(trace, closure, buildPhase, () => TryBuildSeedGroupedDagLongestDepthRows(closure, trace, edges, seeds, MaxDagNodeCount, out rows)))
                {
                    throw new InvalidOperationException("SeedGroupedDagLongestDepthNode requires an acyclic edge relation.");
                }

                context.SeedGroupedDagLongestDepthResults[cacheKey] = rows;
                return rows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private static bool TryBuildProjectGroupedDagReachCountRowsFromDelimitedSources(
            DelimitedRelationSource edgeSource,
            DelimitedRelationSource seedSource,
            int minEdgeCount,
            int minGroupCount,
            int maxNodeCount,
            int maxGroupCount,
            out List<object[]> rows)
        {
            rows = new List<object[]>();
            var globalNodeIds = new Dictionary<string, int>(StringComparer.Ordinal);
            var globalSuccessors = new List<List<int>>();

            int GetGlobalNodeId(string value)
            {
                if (globalNodeIds.TryGetValue(value, out var id))
                {
                    return id;
                }

                id = globalSuccessors.Count;
                globalNodeIds.Add(value, id);
                globalSuccessors.Add(new List<int>());
                return id;
            }

            var edgeCount = 0;
            using (var reader = DelimitedRelationReader.OpenSequentialReader(edgeSource.InputPath))
            {
                for (var i = 0; i < edgeSource.SkipRows; i++)
                {
                    if (reader.ReadLine() is null)
                    {
                        return false;
                    }
                }

                string? line;
                while ((line = reader.ReadLine()) is not null)
                {
                    if (!DelimitedRelationReader.TrySplitTwoColumnLine(line, edgeSource.Delimiter, out var left, out var right))
                    {
                        continue;
                    }

                    var fromId = GetGlobalNodeId(left);
                    var toId = GetGlobalNodeId(right);
                    globalSuccessors[fromId].Add(toId);
                    edgeCount++;
                }
            }

            if (edgeCount < minEdgeCount || globalSuccessors.Count == 0 || globalSuccessors.Count > maxNodeCount)
            {
                return false;
            }

            var groupIds = new Dictionary<string, int>(StringComparer.Ordinal);
            var groupValues = new List<string>();
            var seedPairs = new List<(int GroupId, int NodeId)>();
            using (var reader = DelimitedRelationReader.OpenSequentialReader(seedSource.InputPath))
            {
                for (var i = 0; i < seedSource.SkipRows; i++)
                {
                    if (reader.ReadLine() is null)
                    {
                        return false;
                    }
                }

                string? line;
                while ((line = reader.ReadLine()) is not null)
                {
                    if (!DelimitedRelationReader.TrySplitTwoColumnLine(line, seedSource.Delimiter, out var group, out var node))
                    {
                        continue;
                    }

                    if (!globalNodeIds.TryGetValue(node, out var nodeId))
                    {
                        continue;
                    }

                    if (!groupIds.TryGetValue(group, out var groupId))
                    {
                        groupId = groupValues.Count;
                        if (groupId >= maxGroupCount)
                        {
                            return false;
                        }

                        groupIds.Add(group, groupId);
                        groupValues.Add(group);
                    }

                    seedPairs.Add((groupId, nodeId));
                }
            }

            if (groupValues.Count < minGroupCount || seedPairs.Count == 0)
            {
                return false;
            }

            return TryBuildProjectGroupedDagReachCountRowsFromGraph(globalSuccessors, groupValues, seedPairs, maxNodeCount, out rows);
        }

        private static bool TryBuildProjectGroupedDagReachCountRowsFromGraph(
            List<List<int>> globalSuccessors,
            List<string> groupValues,
            List<(int GroupId, int NodeId)> seedPairs,
            int maxNodeCount,
            out List<object[]> rows)
        {
            rows = new List<object[]>();
            var included = new bool[globalSuccessors.Count];
            var queue = new Queue<int>();
            foreach (var (_, nodeId) in seedPairs)
            {
                if (included[nodeId])
                {
                    continue;
                }

                included[nodeId] = true;
                queue.Enqueue(nodeId);
            }

            while (queue.Count > 0)
            {
                var nodeId = queue.Dequeue();
                foreach (var next in globalSuccessors[nodeId])
                {
                    if (included[next])
                    {
                        continue;
                    }

                    included[next] = true;
                    queue.Enqueue(next);
                }
            }

            var includedCount = 0;
            foreach (var isIncluded in included)
            {
                if (isIncluded)
                {
                    includedCount++;
                }
            }

            if (includedCount == 0 || includedCount > maxNodeCount)
            {
                return false;
            }

            var localIds = new int[included.Length];
            Array.Fill(localIds, -1);
            var localSuccessors = new List<List<int>>(includedCount);
            var localIndegree = new int[includedCount];
            var nextLocalId = 0;

            for (var globalId = 0; globalId < included.Length; globalId++)
            {
                if (!included[globalId])
                {
                    continue;
                }

                localIds[globalId] = nextLocalId;
                localSuccessors.Add(new List<int>());
                nextLocalId++;
            }

            for (var globalId = 0; globalId < included.Length; globalId++)
            {
                if (!included[globalId])
                {
                    continue;
                }

                var fromLocalId = localIds[globalId];
                foreach (var nextGlobalId in globalSuccessors[globalId])
                {
                    if (!included[nextGlobalId])
                    {
                        continue;
                    }

                    var toLocalId = localIds[nextGlobalId];
                    localSuccessors[fromLocalId].Add(toLocalId);
                    localIndegree[toLocalId]++;
                }
            }

            var topoQueue = new Queue<int>();
            for (var i = 0; i < localIndegree.Length; i++)
            {
                if (localIndegree[i] == 0)
                {
                    topoQueue.Enqueue(i);
                }
            }

            var topo = new List<int>(includedCount);
            while (topoQueue.Count > 0)
            {
                var localId = topoQueue.Dequeue();
                topo.Add(localId);
                foreach (var successorLocalId in localSuccessors[localId])
                {
                    localIndegree[successorLocalId]--;
                    if (localIndegree[successorLocalId] == 0)
                    {
                        topoQueue.Enqueue(successorLocalId);
                    }
                }
            }

            if (topo.Count != includedCount)
            {
                return false;
            }

            var wordCount = (groupValues.Count + 63) >> 6;
            var nodeGroupBits = new ulong[includedCount][];
            for (var i = 0; i < includedCount; i++)
            {
                nodeGroupBits[i] = new ulong[wordCount];
            }

            foreach (var (groupId, nodeId) in seedPairs)
            {
                var localNodeId = localIds[nodeId];
                nodeGroupBits[localNodeId][groupId >> 6] |= 1UL << (groupId & 63);
            }

            foreach (var localId in topo)
            {
                var sourceBits = nodeGroupBits[localId];
                if (IsZeroWordArray(sourceBits))
                {
                    continue;
                }

                foreach (var successorLocalId in localSuccessors[localId])
                {
                    OrInto(nodeGroupBits[successorLocalId], sourceBits);
                }
            }

            var counts = new int[groupValues.Count];
            for (var localId = 0; localId < includedCount; localId++)
            {
                var bits = nodeGroupBits[localId];
                for (var wordIndex = 0; wordIndex < bits.Length; wordIndex++)
                {
                    var word = bits[wordIndex];
                    while (word != 0)
                    {
                        var bit = BitOperations.TrailingZeroCount(word);
                        var groupId = (wordIndex << 6) + bit;
                        if (groupId < counts.Length)
                        {
                            counts[groupId]++;
                        }
                        word &= word - 1;
                    }
                }
            }

            rows = new List<object[]>(groupValues.Count);
            for (var groupId = 0; groupId < groupValues.Count; groupId++)
            {
                rows.Add(new object[] { groupValues[groupId], counts[groupId] });
            }

            return true;
        }

        private static bool TryBuildSeedGroupedDagLongestDepthRowsFromDelimitedSources(
            PlanNode traceNode,
            QueryExecutionTrace? trace,
            DelimitedRelationSource edgeSource,
            DelimitedRelationSource seedSource,
            int maxNodeCount,
            out List<object[]> rows)
        {
            rows = new List<object[]>();
            var phaseStopwatch = Stopwatch.StartNew();
            var nodeIds = new Dictionary<string, int>(StringComparer.Ordinal);
            var nodeValues = new List<string>();
            var successors = new List<List<int>>();

            int GetNodeId(string value)
            {
                if (nodeIds.TryGetValue(value, out var id))
                {
                    return id;
                }

                id = successors.Count;
                nodeIds.Add(value, id);
                nodeValues.Add(value);
                successors.Add(new List<int>());
                return id;
            }

            using (var reader = DelimitedRelationReader.OpenSequentialReader(edgeSource.InputPath))
            {
                for (var i = 0; i < edgeSource.SkipRows; i++)
                {
                    if (reader.ReadLine() is null)
                    {
                        return false;
                    }
                }

                string? line;
                while ((line = reader.ReadLine()) is not null)
                {
                    if (!DelimitedRelationReader.TrySplitTwoColumnLine(line, edgeSource.Delimiter, out var left, out var right))
                    {
                        continue;
                    }

                    var fromId = GetNodeId(left);
                    var toId = GetNodeId(right);
                    successors[fromId].Add(toId);
                }
            }

            if (successors.Count == 0 || successors.Count > maxNodeCount)
            {
                return false;
            }
            phaseStopwatch.Stop();
            trace?.RecordPhase(traceNode, "build_graph", phaseStopwatch.Elapsed);

            phaseStopwatch.Restart();
            var groupIds = new Dictionary<string, int>(StringComparer.Ordinal);
            var groupValues = new List<string>();
            var seedPairs = new List<(int GroupId, int NodeId)>();
            using (var reader = DelimitedRelationReader.OpenSequentialReader(seedSource.InputPath))
            {
                for (var i = 0; i < seedSource.SkipRows; i++)
                {
                    if (reader.ReadLine() is null)
                    {
                        return false;
                    }
                }

                string? line;
                while ((line = reader.ReadLine()) is not null)
                {
                    if (!DelimitedRelationReader.TrySplitTwoColumnLine(line, seedSource.Delimiter, out var group, out var node))
                    {
                        continue;
                    }

                    if (!nodeIds.TryGetValue(node, out var nodeId))
                    {
                        continue;
                    }

                    if (!groupIds.TryGetValue(group, out var groupId))
                    {
                        groupId = groupValues.Count;
                        groupIds.Add(group, groupId);
                        groupValues.Add(group);
                    }

                    seedPairs.Add((groupId, nodeId));
                }
            }

            if (seedPairs.Count == 0)
            {
                return true;
            }
            phaseStopwatch.Stop();
            trace?.RecordPhase(traceNode, "seed_grouping", phaseStopwatch.Elapsed);

            return TryBuildSeedGroupedDagLongestDepthRowsFromGraph(traceNode, trace, successors, nodeValues, groupValues, seedPairs, maxNodeCount, out rows);
        }

        private static bool TryBuildSeedGroupedDagLongestDepthRowsFromGraph(
            PlanNode traceNode,
            QueryExecutionTrace? trace,
            List<List<int>> successors,
            List<string> nodeValues,
            List<string> groupValues,
            List<(int GroupId, int NodeId)> seedPairs,
            int maxNodeCount,
            out List<object[]> rows)
        {
            rows = new List<object[]>();
            var phaseStopwatch = Stopwatch.StartNew();
            var reachable = new bool[successors.Count];
            var queue = new Queue<int>(seedPairs.Count);
            foreach (var (_, nodeId) in seedPairs)
            {
                if (reachable[nodeId])
                {
                    continue;
                }

                reachable[nodeId] = true;
                queue.Enqueue(nodeId);
            }

            while (queue.Count > 0)
            {
                var nodeId = queue.Dequeue();
                foreach (var next in successors[nodeId])
                {
                    if (reachable[next])
                    {
                        continue;
                    }

                    reachable[next] = true;
                    queue.Enqueue(next);
                }
            }

            var includedCount = 0;
            foreach (var isReachable in reachable)
            {
                if (isReachable)
                {
                    includedCount++;
                }
            }

            if (includedCount == 0 || includedCount > maxNodeCount)
            {
                return false;
            }
            phaseStopwatch.Stop();
            trace?.RecordPhase(traceNode, "reachable_cone", phaseStopwatch.Elapsed);

            phaseStopwatch.Restart();
            var indegree = new int[successors.Count];
            for (var globalId = 0; globalId < successors.Count; globalId++)
            {
                if (!reachable[globalId])
                {
                    continue;
                }

                foreach (var nextGlobalId in successors[globalId])
                {
                    if (!reachable[nextGlobalId])
                    {
                        continue;
                    }

                    indegree[nextGlobalId]++;
                }
            }

            var topoQueue = new Queue<int>(includedCount);
            for (var i = 0; i < successors.Count; i++)
            {
                if (reachable[i] && indegree[i] == 0)
                {
                    topoQueue.Enqueue(i);
                }
            }

            var topo = new List<int>(includedCount);
            while (topoQueue.Count > 0)
            {
                var nodeId = topoQueue.Dequeue();
                topo.Add(nodeId);
                foreach (var successorId in successors[nodeId])
                {
                    if (!reachable[successorId])
                    {
                        continue;
                    }

                    indegree[successorId]--;
                    if (indegree[successorId] == 0)
                    {
                        topoQueue.Enqueue(successorId);
                    }
                }
            }

            if (topo.Count != includedCount)
            {
                return false;
            }
            phaseStopwatch.Stop();
            trace?.RecordPhase(traceNode, "topological_order", phaseStopwatch.Elapsed);

            phaseStopwatch.Restart();
            var depths = new int[successors.Count];
            for (var i = topo.Count - 1; i >= 0; i--)
            {
                var nodeId = topo[i];
                var best = 1;
                foreach (var successorId in successors[nodeId])
                {
                    if (!reachable[successorId])
                    {
                        continue;
                    }

                    var candidate = depths[successorId] + 1;
                    if (candidate > best)
                    {
                        best = candidate;
                    }
                }

                depths[nodeId] = best;
            }
            phaseStopwatch.Stop();
            trace?.RecordPhase(traceNode, "suffix_depth_dp", phaseStopwatch.Elapsed);

            phaseStopwatch.Restart();
            var groupBest = new int[groupValues.Count];
            foreach (var (groupId, nodeId) in seedPairs)
            {
                if (!reachable[nodeId])
                {
                    continue;
                }

                var depth = depths[nodeId];
                if (depth > groupBest[groupId])
                {
                    groupBest[groupId] = depth;
                }
            }

            rows = new List<object[]>(groupValues.Count);
            for (var groupId = 0; groupId < groupValues.Count; groupId++)
            {
                rows.Add(new object[] { groupValues[groupId], groupBest[groupId] });
            }
            phaseStopwatch.Stop();
            trace?.RecordPhase(traceNode, "group_reduction", phaseStopwatch.Elapsed);

            return true;
        }

        private static bool TryBuildProjectGroupedDagReachRows(
            IReadOnlyList<object[]> edges,
            IReadOnlyList<object[]> seeds,
            int maxNodeCount,
            int maxGroupCount,
            out List<object[]> rows)
        {
            rows = new List<object[]>();
            if (edges.Count == 0 || seeds.Count == 0)
            {
                return false;
            }

            var globalNodeIds = new Dictionary<object?, int>();
            var globalSuccessors = new List<List<int>>();
            var nodeValues = new List<object?>();

            int GetGlobalNodeId(object? value)
            {
                if (globalNodeIds.TryGetValue(value, out var id))
                {
                    return id;
                }

                id = nodeValues.Count;
                globalNodeIds.Add(value, id);
                nodeValues.Add(value);
                globalSuccessors.Add(new List<int>());
                return id;
            }

            foreach (var edge in edges)
            {
                if (edge is null || edge.Length < 2)
                {
                    continue;
                }

                var fromId = GetGlobalNodeId(edge[0]);
                var toId = GetGlobalNodeId(edge[1]);
                globalSuccessors[fromId].Add(toId);
            }

            if (nodeValues.Count == 0 || nodeValues.Count > maxNodeCount)
            {
                return false;
            }

            var globalIndegree = new int[nodeValues.Count];
            for (var i = 0; i < globalSuccessors.Count; i++)
            {
                foreach (var next in globalSuccessors[i])
                {
                    globalIndegree[next]++;
                }
            }

            var groupIds = new Dictionary<object?, int>();
            var groupValues = new List<object?>();
            var seedPairs = new List<(int GroupId, int NodeId)>();
            foreach (var seed in seeds)
            {
                if (seed is null || seed.Length < 2)
                {
                    continue;
                }

                if (!globalNodeIds.TryGetValue(seed[1], out var nodeId))
                {
                    continue;
                }

                if (!groupIds.TryGetValue(seed[0], out var groupId))
                {
                    groupId = groupValues.Count;
                    if (groupId >= maxGroupCount)
                    {
                        return false;
                    }

                    groupIds.Add(seed[0], groupId);
                    groupValues.Add(seed[0]);
                }

                seedPairs.Add((groupId, nodeId));
            }

            if (seedPairs.Count == 0)
            {
                return true;
            }

            var included = new bool[nodeValues.Count];
            var queue = new Queue<int>();
            foreach (var (_, nodeId) in seedPairs)
            {
                if (included[nodeId])
                {
                    continue;
                }

                included[nodeId] = true;
                queue.Enqueue(nodeId);
            }

            while (queue.Count > 0)
            {
                var nodeId = queue.Dequeue();
                foreach (var next in globalSuccessors[nodeId])
                {
                    if (included[next])
                    {
                        continue;
                    }

                    included[next] = true;
                    queue.Enqueue(next);
                }
            }

            var includedCount = 0;
            foreach (var isIncluded in included)
            {
                if (isIncluded)
                {
                    includedCount++;
                }
            }

            if (includedCount == 0 || includedCount > maxNodeCount)
            {
                return false;
            }

            var localIds = new int[nodeValues.Count];
            Array.Fill(localIds, -1);
            var localNodeValues = new object?[includedCount];
            var localSuccessors = new List<List<int>>(includedCount);
            var localIndegree = new int[includedCount];
            var nextLocalId = 0;

            for (var globalId = 0; globalId < included.Length; globalId++)
            {
                if (!included[globalId])
                {
                    continue;
                }

                localIds[globalId] = nextLocalId;
                localNodeValues[nextLocalId] = nodeValues[globalId];
                localSuccessors.Add(new List<int>());
                nextLocalId++;
            }

            for (var globalId = 0; globalId < included.Length; globalId++)
            {
                if (!included[globalId])
                {
                    continue;
                }

                var fromLocalId = localIds[globalId];
                foreach (var nextGlobalId in globalSuccessors[globalId])
                {
                    if (!included[nextGlobalId])
                    {
                        continue;
                    }

                    var toLocalId = localIds[nextGlobalId];
                    localSuccessors[fromLocalId].Add(toLocalId);
                    localIndegree[toLocalId]++;
                }
            }

            var topoQueue = new Queue<int>();
            for (var i = 0; i < localIndegree.Length; i++)
            {
                if (localIndegree[i] == 0)
                {
                    topoQueue.Enqueue(i);
                }
            }

            var topo = new List<int>(includedCount);
            while (topoQueue.Count > 0)
            {
                var localId = topoQueue.Dequeue();
                topo.Add(localId);
                foreach (var successorLocalId in localSuccessors[localId])
                {
                    localIndegree[successorLocalId]--;
                    if (localIndegree[successorLocalId] == 0)
                    {
                        topoQueue.Enqueue(successorLocalId);
                    }
                }
            }

            if (topo.Count != includedCount)
            {
                return false;
            }

            var wordCount = (groupValues.Count + 63) >> 6;
            var nodeGroupBits = new ulong[includedCount][];
            for (var i = 0; i < includedCount; i++)
            {
                nodeGroupBits[i] = new ulong[wordCount];
            }

            foreach (var (groupId, nodeId) in seedPairs)
            {
                var localNodeId = localIds[nodeId];
                nodeGroupBits[localNodeId][groupId >> 6] |= 1UL << (groupId & 63);
            }

            foreach (var localId in topo)
            {
                var sourceBits = nodeGroupBits[localId];
                if (IsZeroWordArray(sourceBits))
                {
                    continue;
                }

                foreach (var successorLocalId in localSuccessors[localId])
                {
                    OrInto(nodeGroupBits[successorLocalId], sourceBits);
                }
            }

            rows = new List<object[]>(Math.Max(includedCount, seedPairs.Count));
            for (var localId = 0; localId < includedCount; localId++)
            {
                var nodeValue = localNodeValues[localId];
                var bits = nodeGroupBits[localId];
                for (var wordIndex = 0; wordIndex < bits.Length; wordIndex++)
                {
                    var word = bits[wordIndex];
                    while (word != 0)
                    {
                        var bit = BitOperations.TrailingZeroCount(word);
                        var groupId = (wordIndex << 6) + bit;
                        if (groupId < groupValues.Count)
                        {
                            rows.Add(new object[] { groupValues[groupId]!, nodeValue! });
                        }

                        word &= word - 1;
                    }
                }
            }

            return true;
        }

        private static bool TryBuildProjectGroupedDagReachCountRows(
            IReadOnlyList<object[]> edges,
            IReadOnlyList<object[]> seeds,
            int maxNodeCount,
            int maxGroupCount,
            out List<object[]> rows)
        {
            rows = new List<object[]>();
            if (edges.Count == 0 || seeds.Count == 0)
            {
                return false;
            }

            var globalNodeIds = new Dictionary<object?, int>();
            var globalSuccessors = new List<List<int>>();

            int GetGlobalNodeId(object? value)
            {
                if (globalNodeIds.TryGetValue(value, out var id))
                {
                    return id;
                }

                id = globalSuccessors.Count;
                globalNodeIds.Add(value, id);
                globalSuccessors.Add(new List<int>());
                return id;
            }

            foreach (var edge in edges)
            {
                if (edge is null || edge.Length < 2)
                {
                    continue;
                }

                var fromId = GetGlobalNodeId(edge[0]);
                var toId = GetGlobalNodeId(edge[1]);
                globalSuccessors[fromId].Add(toId);
            }

            if (globalSuccessors.Count == 0 || globalSuccessors.Count > maxNodeCount)
            {
                return false;
            }

            var groupIds = new Dictionary<object?, int>();
            var groupValues = new List<object?>();
            var seedPairs = new List<(int GroupId, int NodeId)>();
            foreach (var seed in seeds)
            {
                if (seed is null || seed.Length < 2)
                {
                    continue;
                }

                if (!globalNodeIds.TryGetValue(seed[1], out var nodeId))
                {
                    continue;
                }

                if (!groupIds.TryGetValue(seed[0], out var groupId))
                {
                    groupId = groupValues.Count;
                    if (groupId >= maxGroupCount)
                    {
                        return false;
                    }

                    groupIds.Add(seed[0], groupId);
                    groupValues.Add(seed[0]);
                }

                seedPairs.Add((groupId, nodeId));
            }

            if (seedPairs.Count == 0)
            {
                return true;
            }

            var included = new bool[globalSuccessors.Count];
            var queue = new Queue<int>();
            foreach (var (_, nodeId) in seedPairs)
            {
                if (included[nodeId])
                {
                    continue;
                }

                included[nodeId] = true;
                queue.Enqueue(nodeId);
            }

            while (queue.Count > 0)
            {
                var nodeId = queue.Dequeue();
                foreach (var next in globalSuccessors[nodeId])
                {
                    if (included[next])
                    {
                        continue;
                    }

                    included[next] = true;
                    queue.Enqueue(next);
                }
            }

            var includedCount = 0;
            foreach (var isIncluded in included)
            {
                if (isIncluded)
                {
                    includedCount++;
                }
            }

            if (includedCount == 0 || includedCount > maxNodeCount)
            {
                return false;
            }

            var localIds = new int[included.Length];
            Array.Fill(localIds, -1);
            var localSuccessors = new List<List<int>>(includedCount);
            var localIndegree = new int[includedCount];
            var nextLocalId = 0;

            for (var globalId = 0; globalId < included.Length; globalId++)
            {
                if (!included[globalId])
                {
                    continue;
                }

                localIds[globalId] = nextLocalId;
                localSuccessors.Add(new List<int>());
                nextLocalId++;
            }

            for (var globalId = 0; globalId < included.Length; globalId++)
            {
                if (!included[globalId])
                {
                    continue;
                }

                var fromLocalId = localIds[globalId];
                foreach (var nextGlobalId in globalSuccessors[globalId])
                {
                    if (!included[nextGlobalId])
                    {
                        continue;
                    }

                    var toLocalId = localIds[nextGlobalId];
                    localSuccessors[fromLocalId].Add(toLocalId);
                    localIndegree[toLocalId]++;
                }
            }

            var topoQueue = new Queue<int>();
            for (var i = 0; i < localIndegree.Length; i++)
            {
                if (localIndegree[i] == 0)
                {
                    topoQueue.Enqueue(i);
                }
            }

            var topo = new List<int>(includedCount);
            while (topoQueue.Count > 0)
            {
                var localId = topoQueue.Dequeue();
                topo.Add(localId);
                foreach (var successorLocalId in localSuccessors[localId])
                {
                    localIndegree[successorLocalId]--;
                    if (localIndegree[successorLocalId] == 0)
                    {
                        topoQueue.Enqueue(successorLocalId);
                    }
                }
            }

            if (topo.Count != includedCount)
            {
                return false;
            }

            var wordCount = (groupValues.Count + 63) >> 6;
            var nodeGroupBits = new ulong[includedCount][];
            for (var i = 0; i < includedCount; i++)
            {
                nodeGroupBits[i] = new ulong[wordCount];
            }

            foreach (var (groupId, nodeId) in seedPairs)
            {
                var localNodeId = localIds[nodeId];
                nodeGroupBits[localNodeId][groupId >> 6] |= 1UL << (groupId & 63);
            }

            foreach (var localId in topo)
            {
                var sourceBits = nodeGroupBits[localId];
                if (IsZeroWordArray(sourceBits))
                {
                    continue;
                }

                foreach (var successorLocalId in localSuccessors[localId])
                {
                    OrInto(nodeGroupBits[successorLocalId], sourceBits);
                }
            }

            var counts = new int[groupValues.Count];
            for (var localId = 0; localId < includedCount; localId++)
            {
                var bits = nodeGroupBits[localId];
                for (var wordIndex = 0; wordIndex < bits.Length; wordIndex++)
                {
                    var word = bits[wordIndex];
                    while (word != 0)
                    {
                        var bit = BitOperations.TrailingZeroCount(word);
                        var groupId = (wordIndex << 6) + bit;
                        if (groupId < counts.Length)
                        {
                            counts[groupId]++;
                        }

                        word &= word - 1;
                    }
                }
            }

            rows = new List<object[]>(groupValues.Count);
            for (var groupId = 0; groupId < groupValues.Count; groupId++)
            {
                rows.Add(new object[] { groupValues[groupId]!, counts[groupId] });
            }

            return true;
        }

        private static bool TryBuildSeedGroupedDagLongestDepthRows(
            PlanNode traceNode,
            QueryExecutionTrace? trace,
            IReadOnlyList<object[]> edges,
            IReadOnlyList<object[]> seeds,
            int maxNodeCount,
            out List<object[]> rows)
        {
            rows = new List<object[]>();
            if (edges.Count == 0 || seeds.Count == 0)
            {
                return false;
            }

            var phaseStopwatch = Stopwatch.StartNew();
            var nodeIds = new Dictionary<object?, int>();
            var successors = new List<List<int>>();

            int GetNodeId(object? value)
            {
                if (nodeIds.TryGetValue(value, out var id))
                {
                    return id;
                }

                id = successors.Count;
                nodeIds.Add(value, id);
                successors.Add(new List<int>());
                return id;
            }

            foreach (var edge in edges)
            {
                if (edge is null || edge.Length < 2)
                {
                    continue;
                }

                var fromId = GetNodeId(edge[0]);
                var toId = GetNodeId(edge[1]);
                successors[fromId].Add(toId);
            }

            if (successors.Count == 0 || successors.Count > maxNodeCount)
            {
                return false;
            }
            phaseStopwatch.Stop();
            trace?.RecordPhase(traceNode, "build_graph", phaseStopwatch.Elapsed);

            phaseStopwatch.Restart();
            var groupIds = new Dictionary<object?, int>();
            var groupValues = new List<object?>();
            var seedPairs = new List<(int GroupId, int NodeId)>();
            foreach (var seed in seeds)
            {
                if (seed is null || seed.Length < 2)
                {
                    continue;
                }

                if (!nodeIds.TryGetValue(seed[1], out var nodeId))
                {
                    continue;
                }

                if (!groupIds.TryGetValue(seed[0], out var groupId))
                {
                    groupId = groupValues.Count;
                    groupIds.Add(seed[0], groupId);
                    groupValues.Add(seed[0]);
                }

                seedPairs.Add((groupId, nodeId));
            }

            if (seedPairs.Count == 0)
            {
                return true;
            }
            phaseStopwatch.Stop();
            trace?.RecordPhase(traceNode, "seed_grouping", phaseStopwatch.Elapsed);

            phaseStopwatch.Restart();
            var reachable = new bool[successors.Count];
            var queue = new Queue<int>(seedPairs.Count);
            foreach (var (_, nodeId) in seedPairs)
            {
                if (reachable[nodeId])
                {
                    continue;
                }

                reachable[nodeId] = true;
                queue.Enqueue(nodeId);
            }

            while (queue.Count > 0)
            {
                var nodeId = queue.Dequeue();
                foreach (var next in successors[nodeId])
                {
                    if (reachable[next])
                    {
                        continue;
                    }

                    reachable[next] = true;
                    queue.Enqueue(next);
                }
            }

            var includedCount = 0;
            foreach (var isReachable in reachable)
            {
                if (isReachable)
                {
                    includedCount++;
                }
            }

            if (includedCount == 0)
            {
                return true;
            }
            phaseStopwatch.Stop();
            trace?.RecordPhase(traceNode, "reachable_cone", phaseStopwatch.Elapsed);

            phaseStopwatch.Restart();
            var indegree = new int[successors.Count];
            for (var globalId = 0; globalId < successors.Count; globalId++)
            {
                if (!reachable[globalId])
                {
                    continue;
                }

                foreach (var nextGlobalId in successors[globalId])
                {
                    if (!reachable[nextGlobalId])
                    {
                        continue;
                    }

                    indegree[nextGlobalId]++;
                }
            }

            var topoQueue = new Queue<int>(includedCount);
            for (var i = 0; i < successors.Count; i++)
            {
                if (reachable[i] && indegree[i] == 0)
                {
                    topoQueue.Enqueue(i);
                }
            }

            var topo = new List<int>(includedCount);
            while (topoQueue.Count > 0)
            {
                var nodeId = topoQueue.Dequeue();
                topo.Add(nodeId);
                foreach (var successorId in successors[nodeId])
                {
                    if (!reachable[successorId])
                    {
                        continue;
                    }

                    indegree[successorId]--;
                    if (indegree[successorId] == 0)
                    {
                        topoQueue.Enqueue(successorId);
                    }
                }
            }

            if (topo.Count != includedCount)
            {
                return false;
            }
            phaseStopwatch.Stop();
            trace?.RecordPhase(traceNode, "topological_order", phaseStopwatch.Elapsed);

            phaseStopwatch.Restart();
            var depths = new int[successors.Count];
            for (var i = topo.Count - 1; i >= 0; i--)
            {
                var nodeId = topo[i];
                var best = 1;
                foreach (var successorId in successors[nodeId])
                {
                    if (!reachable[successorId])
                    {
                        continue;
                    }

                    var candidate = depths[successorId] + 1;
                    if (candidate > best)
                    {
                        best = candidate;
                    }
                }

                depths[nodeId] = best;
            }
            phaseStopwatch.Stop();
            trace?.RecordPhase(traceNode, "suffix_depth_dp", phaseStopwatch.Elapsed);

            phaseStopwatch.Restart();
            var groupBest = new int[groupValues.Count];
            foreach (var (groupId, nodeId) in seedPairs)
            {
                if (!reachable[nodeId])
                {
                    continue;
                }

                var depth = depths[nodeId];
                if (depth > groupBest[groupId])
                {
                    groupBest[groupId] = depth;
                }
            }

            rows = new List<object[]>(groupValues.Count);
            for (var groupId = 0; groupId < groupValues.Count; groupId++)
            {
                rows.Add(new object[] { groupValues[groupId]!, groupBest[groupId] });
            }
            phaseStopwatch.Stop();
            trace?.RecordPhase(traceNode, "group_reduction", phaseStopwatch.Elapsed);

            return true;
        }

        private static bool IsZeroWordArray(ulong[] words)
        {
            for (var i = 0; i < words.Length; i++)
            {
                if (words[i] != 0UL)
                {
                    return false;
                }
            }

            return true;
        }

        private static void OrInto(ulong[] target, ulong[] source)
        {
            for (var i = 0; i < target.Length; i++)
            {
                target[i] |= source[i];
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
                var singleConcretePairRequest = IsSingleConcreteGroupedPairRequest(targetsBySource, sourcesByTarget);
                var preferForwardSingleProbe = true;
                Dictionary<RowWrapper, List<object[]>>? singleSuccIndex = null;
                Dictionary<RowWrapper, List<object[]>>? singlePredIndex = null;
                var singleForwardCost = 0;
                var singleBackwardCost = 0;

                if (singleConcretePairRequest)
                {
                    var sourceEntry = targetsBySource.First();
                    var sourceKey = sourceEntry.Key.Row;
                    var target = sourceEntry.Value.First();
                    var targetKey = new object[sourceKey.Length];
                    Array.Copy(sourceKey, targetKey, sourceKey.Length - 1);
                    targetKey[sourceKey.Length - 1] = target;

                    var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);
                    var keyCount = closure.GroupIndices.Count;
                    var fromKeyIndices = new int[keyCount + 1];
                    var toKeyIndices = new int[keyCount + 1];
                    for (var i = 0; i < keyCount; i++)
                    {
                        fromKeyIndices[i] = closure.GroupIndices[i];
                        toKeyIndices[i] = closure.GroupIndices[i];
                    }

                    fromKeyIndices[keyCount] = 0;
                    toKeyIndices[keyCount] = 1;
                    singleSuccIndex = GetJoinIndex(closure.EdgeRelation, fromKeyIndices, edges, context);
                    singlePredIndex = GetJoinIndex(closure.EdgeRelation, toKeyIndices, edges, context);
                    singleForwardCost = CountEdgeBucket(singleSuccIndex, sourceKey);
                    singleBackwardCost = CountEdgeBucket(singlePredIndex, targetKey);
                    preferForwardSingleProbe = singleForwardCost <= singleBackwardCost;
                }

                Dictionary<RowWrapper, HashSet<object?>> forwardTargetsBySource = new(wrapperComparer);
                Dictionary<RowWrapper, HashSet<object?>> backwardSourcesByTarget = new(wrapperComparer);
                var canBuildDirectionBatches = !singleConcretePairRequest &&
                    TryBuildGroupedPairProbeDirectionBatches(
                        closure,
                        targetsBySource,
                        context,
                        out forwardTargetsBySource,
                        out backwardSourcesByTarget);
                var canReuseBatchedPairProbeCache =
                    canBuildDirectionBatches &&
                    _pairProbeCacheMaxEntries > 0 &&
                    _pairProbeCacheAdmissionMinCostPerProbe > 0d;
                var canMemoizeForwardBatch =
                    canBuildDirectionBatches &&
                    _cacheContext is not null &&
                    forwardTargetsBySource.Count > 0 &&
                    forwardTargetsBySource.Count <= MaxMemoizedGroupedPairSeeds;
                var canMemoizeBackwardBatch =
                    canBuildDirectionBatches &&
                    _cacheContext is not null &&
                    backwardSourcesByTarget.Count > 0 &&
                    backwardSourcesByTarget.Count <= MaxMemoizedGroupedPairSeeds;
                var canMemoizePairs =
                    !singleConcretePairRequest &&
                    _cacheContext is not null &&
                    targetsBySource.Count <= MaxMemoizedGroupedPairSeeds &&
                    sourcesByTarget.Count <= MaxMemoizedGroupedPairSeeds;
                var preferForwardFallback = targetsBySource.Count <= sourcesByTarget.Count;
                var pairRequestCount = CountClosurePairRequests(targetsBySource.Values);
                IReadOnlyDictionary<ClosurePairPlanStrategy, Func<TimeSpan>>? measurePairStrategyProbes = null;
                if (_enableMeasuredClosurePairStrategy &&
                    !singleConcretePairRequest &&
                    pairRequestCount > 1 &&
                    pairRequestCount <= 16)
                {
                    var probes = new Dictionary<ClosurePairPlanStrategy, Func<TimeSpan>>
                    {
                        [ClosurePairPlanStrategy.Forward] = () => MeasureGroupedClosurePairStrategyProbe(
                            closure,
                            ClosurePairPlanStrategy.Forward,
                            inputPositions,
                            targetsBySource,
                            sourcesByTarget,
                            forwardTargetsBySource,
                            backwardSourcesByTarget,
                            context),
                        [ClosurePairPlanStrategy.Backward] = () => MeasureGroupedClosurePairStrategyProbe(
                            closure,
                            ClosurePairPlanStrategy.Backward,
                            inputPositions,
                            targetsBySource,
                            sourcesByTarget,
                            forwardTargetsBySource,
                            backwardSourcesByTarget,
                            context)
                    };

                    if (canMemoizeForwardBatch || canMemoizePairs)
                    {
                        probes[ClosurePairPlanStrategy.MemoizedBySource] = () => MeasureGroupedClosurePairStrategyProbe(
                            closure,
                            ClosurePairPlanStrategy.MemoizedBySource,
                            inputPositions,
                            targetsBySource,
                            sourcesByTarget,
                            forwardTargetsBySource,
                            backwardSourcesByTarget,
                            context);
                    }

                    if (canMemoizeBackwardBatch || canMemoizePairs)
                    {
                        probes[ClosurePairPlanStrategy.MemoizedByTarget] = () => MeasureGroupedClosurePairStrategyProbe(
                            closure,
                            ClosurePairPlanStrategy.MemoizedByTarget,
                            inputPositions,
                            targetsBySource,
                            sourcesByTarget,
                            forwardTargetsBySource,
                            backwardSourcesByTarget,
                            context);
                    }

                    if (canBuildDirectionBatches &&
                        forwardTargetsBySource.Count > 0 &&
                        backwardSourcesByTarget.Count > 0)
                    {
                        probes[ClosurePairPlanStrategy.MixedDirection] = () => MeasureGroupedClosurePairStrategyProbe(
                            closure,
                            ClosurePairPlanStrategy.MixedDirection,
                            inputPositions,
                            targetsBySource,
                            sourcesByTarget,
                            forwardTargetsBySource,
                            backwardSourcesByTarget,
                            context);

                        if (canReuseBatchedPairProbeCache)
                        {
                            probes[ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache] = () => MeasureGroupedClosurePairStrategyProbe(
                                closure,
                                ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache,
                                inputPositions,
                                targetsBySource,
                                sourcesByTarget,
                                forwardTargetsBySource,
                                backwardSourcesByTarget,
                                context);
                        }
                    }

                    measurePairStrategyProbes = probes;
                }

                var pairStrategySelection = ResolveClosurePairStrategy(
                    trace,
                    closure,
                    _closurePairStrategy,
                    pairRequestCount,
                    targetsBySource.Count,
                    sourcesByTarget.Count,
                    singleConcretePairRequest,
                    preferForwardSingleProbe,
                    canBuildDirectionBatches,
                    forwardTargetsBySource.Count > 0,
                    backwardSourcesByTarget.Count > 0,
                    canReuseBatchedPairProbeCache,
                    canMemoizeForwardBatch,
                    canMemoizeBackwardBatch,
                    canMemoizePairs,
                    preferForwardFallback,
                    _useSeededClosureCachesForPairBatches && _cacheContext is not null,
                    measurePairStrategyProbes);
                RecordClosurePairStrategy(trace, closure, "GroupedTransitiveClosurePairs", pairStrategySelection);

                return pairStrategySelection.Strategy switch
                {
                    ClosurePairPlanStrategy.SingleProbeForward => ExecuteSeededGroupedTransitiveClosurePairsSingleProbe(
                        closure,
                        inputPositions,
                        parameters,
                        targetsBySource,
                        sourcesByTarget,
                        context,
                        preferForward: true,
                        singleSuccIndex!,
                        singlePredIndex!,
                        singleForwardCost,
                        singleBackwardCost),
                    ClosurePairPlanStrategy.SingleProbeBackward => ExecuteSeededGroupedTransitiveClosurePairsSingleProbe(
                        closure,
                        inputPositions,
                        parameters,
                        targetsBySource,
                        sourcesByTarget,
                        context,
                        preferForward: false,
                        singleSuccIndex!,
                        singlePredIndex!,
                        singleForwardCost,
                        singleBackwardCost),
                    ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache => ExecuteSeededGroupedTransitiveClosurePairsMixedDirectionWithPairProbeCache(
                        closure,
                        inputPositions,
                        forwardTargetsBySource,
                        backwardSourcesByTarget,
                        context),
                    ClosurePairPlanStrategy.MixedDirection => ExecuteSeededGroupedTransitiveClosurePairsMixedDirection(
                        closure,
                        inputPositions,
                        forwardTargetsBySource,
                        backwardSourcesByTarget,
                        context),
                    ClosurePairPlanStrategy.MemoizedBySource => ExecuteSeededGroupedTransitiveClosurePairsMemoizedBySource(
                        closure,
                        inputPositions,
                        targetsBySource,
                        context),
                    ClosurePairPlanStrategy.MemoizedByTarget => ExecuteSeededGroupedTransitiveClosurePairsMemoizedByTarget(
                        closure,
                        inputPositions,
                        sourcesByTarget,
                        context),
                    ClosurePairPlanStrategy.Forward => ExecuteSeededGroupedTransitiveClosurePairsForward(
                        closure,
                        targetsBySource,
                        context),
                    _ => ExecuteSeededGroupedTransitiveClosurePairsBackward(
                        closure,
                        sourcesByTarget,
                        context)
                };
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private IEnumerable<object[]> ExecuteSeededGroupedTransitiveClosurePairsSingleProbe(
            GroupedTransitiveClosureNode closure,
            IReadOnlyList<int> inputPositions,
            IReadOnlyList<object[]> parameters,
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> targetsBySource,
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> sourcesByTarget,
            EvaluationContext context,
            bool preferForward,
            Dictionary<RowWrapper, List<object[]>> succIndex,
            Dictionary<RowWrapper, List<object[]>> predIndex,
            int forwardCost,
            int backwardCost)
        {
            var trace = context.Trace;
            var probeCount = ComputeSingleGroupedPairProbeNormalizationCount(parameters, inputPositions, closure.GroupIndices);
            var sourceEntry = targetsBySource.First();
            var sourceKey = sourceEntry.Key.Row;
            var target = sourceEntry.Value.First();
            var targetKey = new object[sourceKey.Length];
            Array.Copy(sourceKey, targetKey, sourceKey.Length - 1);
            targetKey[sourceKey.Length - 1] = target;
            var groupedCacheKey = (closure.EdgeRelation, closure.Predicate, string.Join(",", closure.GroupIndices));
            var groupedPairKey = new object[sourceKey.Length + 1];
            Array.Copy(sourceKey, groupedPairKey, sourceKey.Length);
            groupedPairKey[sourceKey.Length] = target;
            var sourceLabel = string.Join("|", sourceKey.Select(value => value is null ? "<null>" : FormatCacheSeedValue(value)));
            var targetLabel = target is null ? "<null>" : FormatCacheSeedValue(target);
            var traceKey =
                $"{closure.Predicate.Name}/{closure.Predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:groups={string.Join(",", closure.GroupIndices)}:pair={sourceLabel}->{targetLabel}";
            var canReusePairProbeCache = _pairProbeCacheMaxEntries > 0;

            if (canReusePairProbeCache &&
                context.GroupedTransitiveClosurePairProbeResults.TryGetValue(groupedCacheKey, out var cachedByPair) &&
                TryGetLruRowWrapperCacheValue(cachedByPair, new RowWrapper(groupedPairKey), out var pairReachable))
            {
                trace?.RecordCacheLookup("GroupedTransitiveClosurePairsSingleProbe", traceKey, hit: true, built: false);
                if (!pairReachable)
                {
                    return Array.Empty<object[]>();
                }

                return new List<object[]>
                {
                    BuildGroupedClosureRow(closure.Predicate.Arity, closure.GroupIndices.Count, closure.GroupIndices, groupedPairKey)
                };
            }

            trace?.RecordCacheLookup("GroupedTransitiveClosurePairsSingleProbe", traceKey, hit: false, built: true);

            var rows = preferForward
                ? ExecuteSeededGroupedTransitiveClosurePairsForward(closure, targetsBySource, context).ToList()
                : ExecuteSeededGroupedTransitiveClosurePairsBackward(closure, sourcesByTarget, context).ToList();

            if (canReusePairProbeCache)
            {
                if (!context.GroupedTransitiveClosurePairProbeResults.TryGetValue(groupedCacheKey, out var pairStore))
                {
                    pairStore = new Dictionary<RowWrapper, bool>(StructuralRowWrapperComparer);
                    context.GroupedTransitiveClosurePairProbeResults.Add(groupedCacheKey, pairStore);
                }

                var admitPairProbeCache = ShouldAdmitPairProbeCacheEntry(
                    forwardCost,
                    backwardCost,
                    probeCount,
                    useCostPerProbeGate: true);
                TryAdmitLruBoundedRowWrapperCacheEntry(
                    pairStore,
                    new RowWrapper(groupedPairKey),
                    rows.Count > 0,
                    _pairProbeCacheMaxEntries,
                    admitPairProbeCache,
                    trace,
                    "GroupedTransitiveClosurePairsSingleProbe",
                    traceKey);
            }

            return rows;
        }

        private IEnumerable<object[]> ExecuteSeededGroupedTransitiveClosurePairsMixedDirection(
            GroupedTransitiveClosureNode closure,
            IReadOnlyList<int> inputPositions,
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> forwardTargetsBySource,
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> backwardSourcesByTarget,
            EvaluationContext context)
        {
            if (_useSeededClosureCachesForPairBatches && _cacheContext is not null)
            {
                foreach (var row in ExecuteSeededGroupedTransitiveClosurePairsMemoizedBySource(closure, inputPositions, forwardTargetsBySource, context))
                {
                    yield return row;
                }

                foreach (var row in ExecuteSeededGroupedTransitiveClosurePairsMemoizedByTarget(closure, inputPositions, backwardSourcesByTarget, context))
                {
                    yield return row;
                }

                yield break;
            }

            foreach (var row in ExecuteSeededGroupedTransitiveClosurePairsForward(closure, forwardTargetsBySource, context))
            {
                yield return row;
            }

            foreach (var row in ExecuteSeededGroupedTransitiveClosurePairsBackward(closure, backwardSourcesByTarget, context))
            {
                yield return row;
            }
        }

        private IEnumerable<object[]> ExecuteSeededGroupedTransitiveClosurePairsMixedDirectionWithPairProbeCache(
            GroupedTransitiveClosureNode closure,
            IReadOnlyList<int> inputPositions,
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> forwardTargetsBySource,
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> backwardSourcesByTarget,
            EvaluationContext context)
        {
            var trace = context.Trace;
            var groupedCacheKey = (closure.EdgeRelation, closure.Predicate, string.Join(",", closure.GroupIndices));
            if (!context.GroupedTransitiveClosurePairProbeResults.TryGetValue(groupedCacheKey, out var pairStore))
            {
                pairStore = new Dictionary<RowWrapper, bool>(StructuralRowWrapperComparer);
                context.GroupedTransitiveClosurePairProbeResults.Add(groupedCacheKey, pairStore);
            }

            var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);
            var groupKeyCount = closure.GroupIndices.Count;
            var fromKeyIndices = new int[groupKeyCount + 1];
            var toKeyIndices = new int[groupKeyCount + 1];
            for (var i = 0; i < groupKeyCount; i++)
            {
                fromKeyIndices[i] = closure.GroupIndices[i];
                toKeyIndices[i] = closure.GroupIndices[i];
            }

            fromKeyIndices[groupKeyCount] = 0;
            toKeyIndices[groupKeyCount] = 1;

            var succIndex = GetJoinIndex(closure.EdgeRelation, fromKeyIndices, edges, context);
            var predIndex = GetJoinIndex(closure.EdgeRelation, toKeyIndices, edges, context);
            var wrapperComparer = new RowWrapperComparer(StructuralArrayComparer.Instance);
            var sourceCostCache = new Dictionary<RowWrapper, int>(wrapperComparer);
            var targetCostCache = new Dictionary<RowWrapper, int>(wrapperComparer);
            var forwardMisses = new Dictionary<RowWrapper, HashSet<object?>>(wrapperComparer);
            var backwardMisses = new Dictionary<RowWrapper, HashSet<object?>>(wrapperComparer);
            var cachedRows = new List<object[]>();
            var missedPairs = new List<(object[] PairKey, int ForwardCost, int BackwardCost)>();

            foreach (var sourceEntry in forwardTargetsBySource)
            {
                var sourceWrapper = sourceEntry.Key;
                var sourceKey = sourceWrapper.Row;

                if (!sourceCostCache.TryGetValue(sourceWrapper, out var forwardCost))
                {
                    forwardCost = CountEdgeBucket(succIndex, sourceKey);
                    sourceCostCache.Add(sourceWrapper, forwardCost);
                }

                foreach (var target in sourceEntry.Value)
                {
                    var targetKey = new object[sourceKey.Length];
                    Array.Copy(sourceKey, targetKey, sourceKey.Length - 1);
                    targetKey[sourceKey.Length - 1] = target;
                    var targetWrapper = new RowWrapper(targetKey);

                    if (!targetCostCache.TryGetValue(targetWrapper, out var backwardCost))
                    {
                        backwardCost = CountEdgeBucket(predIndex, targetKey);
                        targetCostCache.Add(targetWrapper, backwardCost);
                    }

                    var groupedPairKey = new object[sourceKey.Length + 1];
                    Array.Copy(sourceKey, groupedPairKey, sourceKey.Length);
                    groupedPairKey[sourceKey.Length] = target;
                    var sourceLabel = string.Join("|", sourceKey.Select(value => value is null ? "<null>" : FormatCacheSeedValue(value)));
                    var targetLabel = target is null ? "<null>" : FormatCacheSeedValue(target);
                    var traceKey =
                        $"{closure.Predicate.Name}/{closure.Predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:groups={string.Join(",", closure.GroupIndices)}:pair={sourceLabel}->{targetLabel}";

                    if (TryGetLruRowWrapperCacheValue(pairStore, new RowWrapper(groupedPairKey), out var pairReachable))
                    {
                        trace?.RecordCacheLookup("GroupedTransitiveClosurePairsSingleProbe", traceKey, hit: true, built: false);
                        if (pairReachable)
                        {
                            cachedRows.Add(BuildGroupedClosureRow(closure.Predicate.Arity, groupKeyCount, closure.GroupIndices, groupedPairKey));
                        }
                    }
                    else
                    {
                        trace?.RecordCacheLookup("GroupedTransitiveClosurePairsSingleProbe", traceKey, hit: false, built: true);
                        AddGroupedPairRequest(forwardMisses, sourceWrapper, target);
                        missedPairs.Add((groupedPairKey, forwardCost, backwardCost));
                    }
                }
            }

            foreach (var targetEntry in backwardSourcesByTarget)
            {
                var targetWrapper = targetEntry.Key;
                var targetKey = targetWrapper.Row;
                if (!targetCostCache.TryGetValue(targetWrapper, out var backwardCost))
                {
                    backwardCost = CountEdgeBucket(predIndex, targetKey);
                    targetCostCache.Add(targetWrapper, backwardCost);
                }

                var target = targetKey[groupKeyCount];
                foreach (var source in targetEntry.Value)
                {
                    var sourceKey = new object[targetKey.Length];
                    Array.Copy(targetKey, sourceKey, targetKey.Length - 1);
                    sourceKey[targetKey.Length - 1] = source;
                    var sourceWrapper = new RowWrapper(sourceKey);

                    if (!sourceCostCache.TryGetValue(sourceWrapper, out var forwardCost))
                    {
                        forwardCost = CountEdgeBucket(succIndex, sourceKey);
                        sourceCostCache.Add(sourceWrapper, forwardCost);
                    }

                    var groupedPairKey = new object[sourceKey.Length + 1];
                    Array.Copy(sourceKey, groupedPairKey, sourceKey.Length);
                    groupedPairKey[sourceKey.Length] = target;
                    var sourceLabel = string.Join("|", sourceKey.Select(value => value is null ? "<null>" : FormatCacheSeedValue(value)));
                    var targetLabel = target is null ? "<null>" : FormatCacheSeedValue(target);
                    var traceKey =
                        $"{closure.Predicate.Name}/{closure.Predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:groups={string.Join(",", closure.GroupIndices)}:pair={sourceLabel}->{targetLabel}";

                    if (TryGetLruRowWrapperCacheValue(pairStore, new RowWrapper(groupedPairKey), out var pairReachable))
                    {
                        trace?.RecordCacheLookup("GroupedTransitiveClosurePairsSingleProbe", traceKey, hit: true, built: false);
                        if (pairReachable)
                        {
                            cachedRows.Add(BuildGroupedClosureRow(closure.Predicate.Arity, groupKeyCount, closure.GroupIndices, groupedPairKey));
                        }
                    }
                    else
                    {
                        trace?.RecordCacheLookup("GroupedTransitiveClosurePairsSingleProbe", traceKey, hit: false, built: true);
                        AddGroupedPairRequest(backwardMisses, targetWrapper, source);
                        missedPairs.Add((groupedPairKey, forwardCost, backwardCost));
                    }
                }
            }

            List<object[]> computedRows;
            if (forwardMisses.Count > 0 && backwardMisses.Count > 0)
            {
                computedRows = ExecuteSeededGroupedTransitiveClosurePairsMixedDirection(
                    closure,
                    inputPositions,
                    forwardMisses,
                    backwardMisses,
                    context).ToList();
            }
            else if (forwardMisses.Count > 0)
            {
                computedRows = ExecuteSeededGroupedTransitiveClosurePairsForward(
                    closure,
                    forwardMisses,
                    context).ToList();
            }
            else if (backwardMisses.Count > 0)
            {
                computedRows = ExecuteSeededGroupedTransitiveClosurePairsBackward(
                    closure,
                    backwardMisses,
                    context).ToList();
            }
            else
            {
                computedRows = new List<object[]>();
            }

            var reachablePairs = new HashSet<RowWrapper>(StructuralRowWrapperComparer);
            foreach (var row in computedRows)
            {
                if (row is null || row.Length < 2)
                {
                    continue;
                }

                var pairKey = new object[groupKeyCount + 2];
                for (var i = 0; i < groupKeyCount; i++)
                {
                    pairKey[i] = row[closure.GroupIndices[i]];
                }

                pairKey[groupKeyCount] = row[0];
                pairKey[groupKeyCount + 1] = row[1];
                reachablePairs.Add(new RowWrapper(pairKey));
            }

            var probeCount = ComputeMixedGroupedPairProbeNormalizationCount(missedPairs, groupKeyCount);
            foreach (var pair in missedPairs)
            {
                var pairKey = pair.PairKey;
                var sourceSlice = new object[groupKeyCount + 1];
                Array.Copy(pairKey, sourceSlice, groupKeyCount + 1);
                var target = pairKey[groupKeyCount + 1];
                var sourceLabel = string.Join("|", sourceSlice.Select(value => value is null ? "<null>" : FormatCacheSeedValue(value)));
                var targetLabel = target is null ? "<null>" : FormatCacheSeedValue(target);
                var traceKey =
                    $"{closure.Predicate.Name}/{closure.Predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:groups={string.Join(",", closure.GroupIndices)}:pair={sourceLabel}->{targetLabel}";
                var admitPairProbeCache = ShouldAdmitPairProbeCacheEntry(
                    pair.ForwardCost,
                    pair.BackwardCost,
                    probeCount,
                    useCostPerProbeGate: true);
                var pairReachable = reachablePairs.Contains(new RowWrapper(pairKey));

                TryAdmitLruBoundedRowWrapperCacheEntry(
                    pairStore,
                    new RowWrapper(pairKey),
                    pairReachable,
                    _pairProbeCacheMaxEntries,
                    admitPairProbeCache,
                    trace,
                    "GroupedTransitiveClosurePairsSingleProbe",
                    traceKey);
            }

            if (cachedRows.Count == 0)
            {
                return computedRows;
            }

            if (computedRows.Count == 0)
            {
                return cachedRows;
            }

            cachedRows.AddRange(computedRows);
            return cachedRows;
        }

        private bool TryBuildGroupedPairProbeDirectionBatches(
            GroupedTransitiveClosureNode closure,
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> targetsBySource,
            EvaluationContext context,
            out Dictionary<RowWrapper, HashSet<object?>> forwardTargetsBySource,
            out Dictionary<RowWrapper, HashSet<object?>> backwardSourcesByTarget)
        {
            var wrapperComparer = new RowWrapperComparer(StructuralArrayComparer.Instance);
            forwardTargetsBySource = new Dictionary<RowWrapper, HashSet<object?>>(wrapperComparer);
            backwardSourcesByTarget = new Dictionary<RowWrapper, HashSet<object?>>(wrapperComparer);

            if (!IsMultiConcreteGroupedPairRequest(targetsBySource, closure.GroupIndices.Count))
            {
                return false;
            }

            var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);
            var groupKeyCount = closure.GroupIndices.Count;
            var fromKeyIndices = new int[groupKeyCount + 1];
            var toKeyIndices = new int[groupKeyCount + 1];
            for (var i = 0; i < groupKeyCount; i++)
            {
                fromKeyIndices[i] = closure.GroupIndices[i];
                toKeyIndices[i] = closure.GroupIndices[i];
            }

            fromKeyIndices[groupKeyCount] = 0;
            toKeyIndices[groupKeyCount] = 1;

            var succIndex = GetJoinIndex(closure.EdgeRelation, fromKeyIndices, edges, context);
            var predIndex = GetJoinIndex(closure.EdgeRelation, toKeyIndices, edges, context);
            var sourceCostCache = new Dictionary<RowWrapper, int>(wrapperComparer);
            var targetCostCache = new Dictionary<RowWrapper, int>(wrapperComparer);

            foreach (var sourceEntry in targetsBySource)
            {
                var sourceWrapper = sourceEntry.Key;
                var sourceKey = sourceWrapper.Row;

                if (!sourceCostCache.TryGetValue(sourceWrapper, out var forwardCost))
                {
                    forwardCost = CountEdgeBucket(succIndex, sourceKey);
                    sourceCostCache.Add(sourceWrapper, forwardCost);
                }

                var source = sourceKey[groupKeyCount];
                foreach (var target in sourceEntry.Value)
                {
                    if (!IsConcretePairTarget(target))
                    {
                        forwardTargetsBySource.Clear();
                        backwardSourcesByTarget.Clear();
                        return false;
                    }

                    var targetKey = new object[sourceKey.Length];
                    Array.Copy(sourceKey, targetKey, sourceKey.Length - 1);
                    targetKey[sourceKey.Length - 1] = target;
                    var targetWrapper = new RowWrapper(targetKey);

                    if (!targetCostCache.TryGetValue(targetWrapper, out var backwardCost))
                    {
                        backwardCost = CountEdgeBucket(predIndex, targetKey);
                        targetCostCache.Add(targetWrapper, backwardCost);
                    }

                    if (forwardCost <= backwardCost)
                    {
                        AddGroupedPairRequest(forwardTargetsBySource, sourceWrapper, target);
                    }
                    else
                    {
                        AddGroupedPairRequest(backwardSourcesByTarget, targetWrapper, source);
                    }
                }
            }

            return forwardTargetsBySource.Count > 0 || backwardSourcesByTarget.Count > 0;
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
                var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);

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
                var canReuseSeededCache = _seededCacheMaxEntries > 0;

                if (canReuseSeededCache &&
                    context.GroupedTransitiveClosureSeededResults.TryGetValue(cacheKey, out var cachedBySeed) &&
                    TryGetLruRowWrapperCacheValue(cachedBySeed, new RowWrapper(flatSeedKey), out var cachedRows))
                {
                    trace?.RecordCacheLookup("GroupedTransitiveClosureSeeded", traceKey, hit: true, built: false);
                    return cachedRows;
                }

                trace?.RecordCacheLookup("GroupedTransitiveClosureSeeded", traceKey, hit: false, built: true);

                const int MaxMemoizedGroupedSeedCount = 32;
                var canMemoizeSeeds =
                    _cacheContext is not null &&
                    orderedSeeds.Count > 1 &&
                    orderedSeeds.Count <= MaxMemoizedGroupedSeedCount &&
                    orderedSeeds.TrueForAll(seedKey => !HasWildcardSeedComponents(seedKey, groupCount));

                if (canMemoizeSeeds)
                {
                    trace?.RecordStrategy(closure, "GroupedTransitiveClosureSeededMemoizedMulti");
                    var memoizedRows = new List<object[]>();

                    foreach (var seedKey in orderedSeeds)
                    {
                        var seedParams = new List<object[]>(1)
                        {
                            BuildGroupedSeedTuple(closure, inputPositions, seedKey, seedPosition: 0)
                        };

                        foreach (var row in ExecuteSeededGroupedTransitiveClosureBySource(closure, inputPositions, seedParams, context))
                        {
                            memoizedRows.Add(row);
                        }
                    }

                    if (canReuseSeededCache)
                    {
                        if (!context.GroupedTransitiveClosureSeededResults.TryGetValue(cacheKey, out var memoizedStoreBySeed))
                        {
                            memoizedStoreBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                            context.GroupedTransitiveClosureSeededResults.Add(cacheKey, memoizedStoreBySeed);
                        }

                        var admitSeededCache = ShouldAdmitSeededCacheRows(
                            memoizedRows.Count,
                            orderedSeeds.Count,
                            useRowsPerSeedGate: true);
                        TryAdmitLruBoundedRowWrapperCacheEntry(
                            memoizedStoreBySeed,
                            new RowWrapper(flatSeedKey),
                            memoizedRows,
                            _seededCacheMaxEntries,
                            admitSeededCache,
                            trace,
                            "GroupedTransitiveClosureSeeded",
                            traceKey);
                    }

                    return memoizedRows;
                }

                if (orderedSeeds.Count == 1 && !HasWildcardSeedComponents(orderedSeeds[0], groupCount))
                {
                    trace?.RecordStrategy(closure, "GroupedTransitiveClosureSeededSingle");

                    var seedKey = orderedSeeds[0];
                    var visitedNodes = new HashSet<object?>();
                    var singleRows = new List<object[]>();
                    var deltaNodes = new List<object?>();

                    var initialLookup = new object[groupCount + 1];
                    Array.Copy(seedKey, initialLookup, initialLookup.Length);
                    if (succIndex.TryGetValue(new RowWrapper(initialLookup), out var bucket))
                    {
                        foreach (var edge in bucket)
                        {
                            if (edge is null || edge.Length <= maxRequiredIndex)
                            {
                                continue;
                            }

                            var to = edge[1];
                            if (!visitedNodes.Add(to))
                            {
                                continue;
                            }

                            var key = new object[groupCount + 2];
                            Array.Copy(seedKey, key, seedKey.Length);
                            key[groupCount + 1] = to;
                            singleRows.Add(BuildGroupedClosureRow(width, groupCount, closure.GroupIndices, key));
                            deltaNodes.Add(to);
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
                            var lookup = new object[groupCount + 1];
                            Array.Copy(seedKey, lookup, groupCount);
                            lookup[groupCount] = node;

                            if (!succIndex.TryGetValue(new RowWrapper(lookup), out bucket))
                            {
                                continue;
                            }

                            foreach (var edge in bucket)
                            {
                                if (edge is null || edge.Length <= maxRequiredIndex)
                                {
                                    continue;
                                }

                                var next = edge[1];
                                if (!visitedNodes.Add(next))
                                {
                                    continue;
                                }

                                var nextKey = new object[groupCount + 2];
                                Array.Copy(seedKey, nextKey, seedKey.Length);
                                nextKey[groupCount + 1] = next;
                                singleRows.Add(BuildGroupedClosureRow(width, groupCount, closure.GroupIndices, nextKey));
                                nextDeltaNodes.Add(next);
                            }
                        }

                        deltaNodes = nextDeltaNodes;
                        trace?.RecordFixpointIteration(closure, predicate, singleIteration, deltaNodes.Count, singleRows.Count);
                    }

                    if (canReuseSeededCache)
                    {
                        if (!context.GroupedTransitiveClosureSeededResults.TryGetValue(cacheKey, out var singleStoreBySeed))
                        {
                            singleStoreBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                            context.GroupedTransitiveClosureSeededResults.Add(cacheKey, singleStoreBySeed);
                        }

                        var admitSeededCache = ShouldAdmitSeededCacheRows(
                            singleRows.Count,
                            orderedSeeds.Count,
                            useRowsPerSeedGate: true);
                        TryAdmitLruBoundedRowWrapperCacheEntry(
                            singleStoreBySeed,
                            new RowWrapper(flatSeedKey),
                            singleRows,
                            _seededCacheMaxEntries,
                            admitSeededCache,
                            trace,
                            "GroupedTransitiveClosureSeeded",
                            traceKey);
                    }

                    return singleRows;
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

                if (canReuseSeededCache)
                {
                    if (!context.GroupedTransitiveClosureSeededResults.TryGetValue(cacheKey, out var storeBySeed))
                    {
                        storeBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                        context.GroupedTransitiveClosureSeededResults.Add(cacheKey, storeBySeed);
                    }

                    var admitSeededCache = ShouldAdmitSeededCacheRows(
                        totalRows.Count,
                        orderedSeeds.Count,
                        useRowsPerSeedGate: true);
                    TryAdmitLruBoundedRowWrapperCacheEntry(
                        storeBySeed,
                        new RowWrapper(flatSeedKey),
                        totalRows,
                        _seededCacheMaxEntries,
                        admitSeededCache,
                        trace,
                        "GroupedTransitiveClosureSeeded",
                        traceKey);
                }

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
                var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);

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
                var canReuseSeededCache = _seededCacheMaxEntries > 0;

                if (canReuseSeededCache &&
                    context.GroupedTransitiveClosureSeededByTargetResults.TryGetValue(cacheKey, out var cachedBySeed) &&
                    TryGetLruRowWrapperCacheValue(cachedBySeed, new RowWrapper(flatSeedKey), out var cachedRows))
                {
                    trace?.RecordCacheLookup("GroupedTransitiveClosureSeededByTarget", traceKey, hit: true, built: false);
                    return cachedRows;
                }

                trace?.RecordCacheLookup("GroupedTransitiveClosureSeededByTarget", traceKey, hit: false, built: true);

                const int MaxMemoizedGroupedTargetSeedCount = 32;
                var canMemoizeSeeds =
                    _cacheContext is not null &&
                    orderedSeeds.Count > 1 &&
                    orderedSeeds.Count <= MaxMemoizedGroupedTargetSeedCount &&
                    orderedSeeds.TrueForAll(seedKey => !HasWildcardSeedComponents(seedKey, groupCount));

                if (canMemoizeSeeds)
                {
                    trace?.RecordStrategy(closure, "GroupedTransitiveClosureSeededByTargetMemoizedMulti");
                    var memoizedRows = new List<object[]>();

                    foreach (var seedKey in orderedSeeds)
                    {
                        var seedParams = new List<object[]>(1)
                        {
                            BuildGroupedSeedTuple(closure, inputPositions, seedKey, seedPosition: 1)
                        };

                        foreach (var row in ExecuteSeededGroupedTransitiveClosureByTarget(closure, inputPositions, seedParams, context))
                        {
                            memoizedRows.Add(row);
                        }
                    }

                    if (canReuseSeededCache)
                    {
                        if (!context.GroupedTransitiveClosureSeededByTargetResults.TryGetValue(cacheKey, out var memoizedStoreBySeed))
                        {
                            memoizedStoreBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                            context.GroupedTransitiveClosureSeededByTargetResults.Add(cacheKey, memoizedStoreBySeed);
                        }

                        var admitSeededCache = ShouldAdmitSeededCacheRows(
                            memoizedRows.Count,
                            orderedSeeds.Count,
                            useRowsPerSeedGate: true);
                        TryAdmitLruBoundedRowWrapperCacheEntry(
                            memoizedStoreBySeed,
                            new RowWrapper(flatSeedKey),
                            memoizedRows,
                            _seededCacheMaxEntries,
                            admitSeededCache,
                            trace,
                            "GroupedTransitiveClosureSeededByTarget",
                            traceKey);
                    }

                    return memoizedRows;
                }

                if (orderedSeeds.Count == 1 && !HasWildcardSeedComponents(orderedSeeds[0], groupCount))
                {
                    trace?.RecordStrategy(closure, "GroupedTransitiveClosureSeededByTargetSingle");

                    var seedKey = orderedSeeds[0];
                    var target = seedKey[groupCount];
                    var visitedNodes = new HashSet<object?>();
                    var singleRows = new List<object[]>();
                    var deltaNodes = new List<object?>();

                    var initialLookup = new object[groupCount + 1];
                    Array.Copy(seedKey, initialLookup, initialLookup.Length);
                    if (predIndex.TryGetValue(new RowWrapper(initialLookup), out var bucket))
                    {
                        foreach (var edge in bucket)
                        {
                            if (edge is null || edge.Length <= maxRequiredIndex)
                            {
                                continue;
                            }

                            var from = edge[0];
                            if (!visitedNodes.Add(from))
                            {
                                continue;
                            }

                            var key = new object[groupCount + 2];
                            Array.Copy(seedKey, key, groupCount);
                            key[groupCount] = from;
                            key[groupCount + 1] = target;
                            singleRows.Add(BuildGroupedClosureRow(width, groupCount, closure.GroupIndices, key));
                            deltaNodes.Add(from);
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
                            var lookup = new object[groupCount + 1];
                            Array.Copy(seedKey, lookup, groupCount);
                            lookup[groupCount] = node;

                            if (!predIndex.TryGetValue(new RowWrapper(lookup), out bucket))
                            {
                                continue;
                            }

                            foreach (var edge in bucket)
                            {
                                if (edge is null || edge.Length <= maxRequiredIndex)
                                {
                                    continue;
                                }

                                var prev = edge[0];
                                if (!visitedNodes.Add(prev))
                                {
                                    continue;
                                }

                                var nextKey = new object[groupCount + 2];
                                Array.Copy(seedKey, nextKey, groupCount);
                                nextKey[groupCount] = prev;
                                nextKey[groupCount + 1] = target;
                                singleRows.Add(BuildGroupedClosureRow(width, groupCount, closure.GroupIndices, nextKey));
                                nextDeltaNodes.Add(prev);
                            }
                        }

                        deltaNodes = nextDeltaNodes;
                        trace?.RecordFixpointIteration(closure, predicate, singleIteration, deltaNodes.Count, singleRows.Count);
                    }

                    if (canReuseSeededCache)
                    {
                        if (!context.GroupedTransitiveClosureSeededByTargetResults.TryGetValue(cacheKey, out var singleStoreBySeed))
                        {
                            singleStoreBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                            context.GroupedTransitiveClosureSeededByTargetResults.Add(cacheKey, singleStoreBySeed);
                        }

                        var admitSeededCache = ShouldAdmitSeededCacheRows(
                            singleRows.Count,
                            orderedSeeds.Count,
                            useRowsPerSeedGate: true);
                        TryAdmitLruBoundedRowWrapperCacheEntry(
                            singleStoreBySeed,
                            new RowWrapper(flatSeedKey),
                            singleRows,
                            _seededCacheMaxEntries,
                            admitSeededCache,
                            trace,
                            "GroupedTransitiveClosureSeededByTarget",
                            traceKey);
                    }

                    return singleRows;
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

                if (canReuseSeededCache)
                {
                    if (!context.GroupedTransitiveClosureSeededByTargetResults.TryGetValue(cacheKey, out var storeBySeed))
                    {
                        storeBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                        context.GroupedTransitiveClosureSeededByTargetResults.Add(cacheKey, storeBySeed);
                    }

                    var admitSeededCache = ShouldAdmitSeededCacheRows(
                        totalRows.Count,
                        orderedSeeds.Count,
                        useRowsPerSeedGate: true);
                    TryAdmitLruBoundedRowWrapperCacheEntry(
                        storeBySeed,
                        new RowWrapper(flatSeedKey),
                        totalRows,
                        _seededCacheMaxEntries,
                        admitSeededCache,
                        trace,
                        "GroupedTransitiveClosureSeededByTarget",
                        traceKey);
                }

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

        private static object[] BuildGroupedSeedTuple(
            GroupedTransitiveClosureNode closure,
            IReadOnlyList<int> inputPositions,
            object[] key,
            int seedPosition)
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
            var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);

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
            var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);

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

        private static bool HasWildcardSeedComponents(object[] seedKey, int groupCount)
        {
            for (var i = 0; i < groupCount; i++)
            {
                if (ReferenceEquals(seedKey[i], Wildcard.Value))
                {
                    return true;
                }
            }

            return false;
        }

        private static bool IsSingleConcretePairRequest(
            IReadOnlyDictionary<object?, HashSet<object?>> bySource,
            IReadOnlyDictionary<object?, HashSet<object?>> byTarget)
        {
            if (bySource.Count != 1 || byTarget.Count != 1)
            {
                return false;
            }

            var sourceEntry = bySource.First();
            if (sourceEntry.Value.Count != 1)
            {
                return false;
            }

            var target = sourceEntry.Value.First();
            if (target is null)
            {
                return false;
            }

            if (!byTarget.TryGetValue(target, out var sources) || sources.Count != 1)
            {
                return false;
            }

            return sources.Contains(sourceEntry.Key);
        }

        private static bool IsMultiConcretePairRequest(IReadOnlyDictionary<object?, HashSet<object?>> bySource)
        {
            if (bySource.Count <= 1)
            {
                return false;
            }

            var pairCount = 0;

            foreach (var sourceEntry in bySource)
            {
                if (ReferenceEquals(sourceEntry.Key, Wildcard.Value))
                {
                    return false;
                }

                foreach (var target in sourceEntry.Value)
                {
                    if (!IsConcretePairTarget(target))
                    {
                        return false;
                    }

                    pairCount++;
                }
            }

            return pairCount > 1;
        }

        private static bool IsSingleConcreteGroupedPairRequest(
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> targetsBySource,
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> sourcesByTarget)
        {
            if (targetsBySource.Count != 1 || sourcesByTarget.Count != 1)
            {
                return false;
            }

            var sourceEntry = targetsBySource.First();
            if (sourceEntry.Value.Count != 1)
            {
                return false;
            }

            var target = sourceEntry.Value.First();
            if (target is null)
            {
                return false;
            }

            var sourceKey = sourceEntry.Key.Row;
            if (sourceKey.Length == 0)
            {
                return false;
            }

            var expectedTargetKey = new object[sourceKey.Length];
            Array.Copy(sourceKey, expectedTargetKey, sourceKey.Length - 1);
            expectedTargetKey[sourceKey.Length - 1] = target;

            if (!sourcesByTarget.TryGetValue(new RowWrapper(expectedTargetKey), out var sources) || sources.Count != 1)
            {
                return false;
            }

            return sources.Contains(sourceKey[sourceKey.Length - 1]);
        }

        private static bool IsMultiConcreteGroupedPairRequest(
            IReadOnlyDictionary<RowWrapper, HashSet<object?>> targetsBySource,
            int groupCount)
        {
            if (targetsBySource.Count <= 1)
            {
                return false;
            }

            var pairCount = 0;

            foreach (var sourceEntry in targetsBySource)
            {
                var sourceKey = sourceEntry.Key.Row;
                if (sourceKey.Length <= groupCount)
                {
                    return false;
                }

                if (HasWildcardSeedComponents(sourceKey, groupCount) ||
                    ReferenceEquals(sourceKey[groupCount], Wildcard.Value))
                {
                    return false;
                }

                foreach (var target in sourceEntry.Value)
                {
                    if (!IsConcretePairTarget(target))
                    {
                        return false;
                    }

                    pairCount++;
                }
            }

            return pairCount > 1;
        }

        private static bool IsConcretePairTarget(object? target) =>
            target is not null && !ReferenceEquals(target, Wildcard.Value);

        private static void AddPairRequest(
            Dictionary<object?, HashSet<object?>> valuesByKey,
            object? key,
            object? value)
        {
            if (!valuesByKey.TryGetValue(key, out var values))
            {
                values = new HashSet<object?>();
                valuesByKey.Add(key, values);
            }

            values.Add(value);
        }

        private static void AddGroupedPairRequest(
            Dictionary<RowWrapper, HashSet<object?>> valuesByKey,
            RowWrapper key,
            object? value)
        {
            if (!valuesByKey.TryGetValue(key, out var values))
            {
                values = new HashSet<object?>();
                valuesByKey.Add(key, values);
            }

            values.Add(value);
        }

        private static int CountEdgeBucket(Dictionary<object, List<object[]>> index, object? key)
        {
            var lookupKey = key ?? NullFactIndexKey;
            return index.TryGetValue(lookupKey, out var bucket) ? bucket.Count : 0;
        }

        private static int CountEdgeBucket(Dictionary<RowWrapper, List<object[]>> index, object[] key) =>
            index.TryGetValue(new RowWrapper(key), out var bucket) ? bucket.Count : 0;

        private sealed class LruAccessState
        {
            public long NextAccessId { get; set; }
            public Dictionary<RowWrapper, long> LastAccessByKey { get; } = new(StructuralRowWrapperComparer);
        }

        private static readonly ConditionalWeakTable<object, LruAccessState> LruAccessByStore = new();

        private static LruAccessState GetLruAccessState<TValue>(Dictionary<RowWrapper, TValue> store)
        {
            var state = LruAccessByStore.GetOrCreateValue(store);
            if (store.Count == 0 && state.LastAccessByKey.Count > 0)
            {
                state.LastAccessByKey.Clear();
            }

            return state;
        }

        private static long TouchLruAccess<TValue>(Dictionary<RowWrapper, TValue> store, RowWrapper key)
        {
            var state = GetLruAccessState(store);
            var nextAccessId = state.NextAccessId + 1;
            state.NextAccessId = nextAccessId;
            state.LastAccessByKey[key] = nextAccessId;
            return nextAccessId;
        }

        private static void RemoveLruAccess<TValue>(Dictionary<RowWrapper, TValue> store, RowWrapper key)
        {
            var state = GetLruAccessState(store);
            state.LastAccessByKey.Remove(key);
        }

        private static bool TryGetLruRowWrapperCacheValue<TValue>(
            Dictionary<RowWrapper, TValue> store,
            RowWrapper key,
            out TValue value)
        {
            if (!store.TryGetValue(key, out value))
            {
                value = default!;
                return false;
            }

            // Refresh recency using explicit access timestamps.
            TouchLruAccess(store, key);
            return true;
        }

        private static void SetLruBoundedRowWrapperCacheEntry<TValue>(
            Dictionary<RowWrapper, TValue> store,
            RowWrapper key,
            TValue value,
            int maxEntries,
            QueryExecutionTrace? trace,
            string? cacheName,
            string? traceKey)
        {
            if (maxEntries <= 0)
            {
                return;
            }

            if (store.ContainsKey(key))
            {
                // Update value and refresh recency.
                store[key] = value;
                TouchLruAccess(store, key);
                return;
            }

            while (store.Count >= maxEntries)
            {
                var state = GetLruAccessState(store);
                RowWrapper oldestKey = default!;
                var oldestAccess = long.MaxValue;
                var foundOldestKey = false;

                foreach (var existingKey in store.Keys)
                {
                    var access = state.LastAccessByKey.TryGetValue(existingKey, out var knownAccess) ? knownAccess : 0;
                    if (!foundOldestKey || access < oldestAccess)
                    {
                        oldestKey = existingKey;
                        oldestAccess = access;
                        foundOldestKey = true;
                    }
                }

                if (!foundOldestKey)
                {
                    break;
                }

                store.Remove(oldestKey);
                RemoveLruAccess(store, oldestKey);
                if (trace is not null && cacheName is not null && traceKey is not null)
                {
                    trace.RecordCacheEviction(cacheName, traceKey);
                }
            }

            store.Add(key, value);
            TouchLruAccess(store, key);
        }

        private static void TryAdmitLruBoundedRowWrapperCacheEntry<TValue>(
            Dictionary<RowWrapper, TValue> store,
            RowWrapper key,
            TValue value,
            int maxEntries,
            bool admitted,
            QueryExecutionTrace? trace,
            string cacheName,
            string traceKey)
        {
            trace?.RecordCacheAdmission(cacheName, traceKey, admitted);
            if (!admitted)
            {
                return;
            }

            SetLruBoundedRowWrapperCacheEntry(
                store,
                key,
                value,
                maxEntries,
                trace,
                cacheName,
                traceKey);
        }

        private bool ShouldAdmitSeededCacheRows(
            int rowCount,
            int seedCount = 1,
            bool useRowsPerSeedGate = false)
        {
            if (rowCount < _seededCacheAdmissionMinRows)
            {
                return false;
            }

            if (!useRowsPerSeedGate || _seededCacheAdmissionMinRowsPerSeed <= 0d)
            {
                return true;
            }

            var normalizedSeedCount = Math.Max(1, seedCount);
            var rowsPerSeed = (double)rowCount / normalizedSeedCount;
            return rowsPerSeed >= _seededCacheAdmissionMinRowsPerSeed;
        }

        private bool ShouldAdmitPairProbeCacheEntry(
            int forwardCost,
            int backwardCost,
            int probeCount = 1,
            bool useCostPerProbeGate = false)
        {
            var minCost = Math.Min(Math.Max(0, forwardCost), Math.Max(0, backwardCost));
            if (minCost < _pairProbeCacheAdmissionMinCost)
            {
                return false;
            }

            if (!useCostPerProbeGate || _pairProbeCacheAdmissionMinCostPerProbe <= 0d)
            {
                return true;
            }

            var normalizedProbeCount = Math.Max(1, probeCount);
            var costPerProbe = (double)minCost / normalizedProbeCount;
            return costPerProbe >= _pairProbeCacheAdmissionMinCostPerProbe;
        }

        private static int ComputeMixedPairProbeNormalizationCount(
            IReadOnlyList<(object? Source, object? Target, int ForwardCost, int BackwardCost)> missedPairs)
        {
            if (missedPairs.Count == 0)
            {
                return 1;
            }

            var wrapperComparer = new RowWrapperComparer(StructuralArrayComparer.Instance);
            var distinctSources = new HashSet<RowWrapper>(wrapperComparer);
            var distinctTargets = new HashSet<RowWrapper>(wrapperComparer);

            foreach (var pair in missedPairs)
            {
                distinctSources.Add(new RowWrapper(new object[] { pair.Source! }));
                distinctTargets.Add(new RowWrapper(new object[] { pair.Target! }));
            }

            var distinctProbeKeys = Math.Max(1, distinctSources.Count + distinctTargets.Count);
            return Math.Max(missedPairs.Count, distinctProbeKeys);
        }

        private static int ComputeMixedGroupedPairProbeNormalizationCount(
            IReadOnlyList<(object[] PairKey, int ForwardCost, int BackwardCost)> missedPairs,
            int groupKeyCount)
        {
            if (missedPairs.Count == 0)
            {
                return 1;
            }

            var wrapperComparer = new RowWrapperComparer(StructuralArrayComparer.Instance);
            var distinctSources = new HashSet<RowWrapper>(wrapperComparer);
            var distinctTargets = new HashSet<RowWrapper>(wrapperComparer);

            foreach (var pair in missedPairs)
            {
                var pairKey = pair.PairKey;

                var sourceSlice = new object[groupKeyCount + 1];
                Array.Copy(pairKey, sourceSlice, groupKeyCount + 1);
                distinctSources.Add(new RowWrapper(sourceSlice));

                var targetSlice = new object[groupKeyCount + 1];
                Array.Copy(pairKey, targetSlice, groupKeyCount);
                targetSlice[groupKeyCount] = pairKey[groupKeyCount + 1];
                distinctTargets.Add(new RowWrapper(targetSlice));
            }

            var distinctProbeKeys = Math.Max(1, distinctSources.Count + distinctTargets.Count);
            return Math.Max(missedPairs.Count, distinctProbeKeys);
        }

        private static int ComputeSinglePairProbeNormalizationCount(IReadOnlyList<object[]> parameters)
        {
            if (parameters.Count == 0)
            {
                return 1;
            }

            var seenPairs = new HashSet<RowWrapper>(StructuralRowWrapperComparer);
            foreach (var paramTuple in parameters)
            {
                if (paramTuple is null || paramTuple.Length < 2)
                {
                    continue;
                }

                var source = paramTuple[0];
                var target = paramTuple[1];
                if (!IsConcretePairTarget(target))
                {
                    continue;
                }

                seenPairs.Add(new RowWrapper(new object[] { source!, target! }));
            }

            return Math.Max(1, seenPairs.Count);
        }

        private static int ComputeSingleGroupedPairProbeNormalizationCount(
            IReadOnlyList<object[]> parameters,
            IReadOnlyList<int> inputPositions,
            IReadOnlyList<int> groupIndices)
        {
            if (parameters.Count == 0)
            {
                return 1;
            }

            var groupCount = groupIndices.Count;
            var seenPairs = new HashSet<RowWrapper>(StructuralRowWrapperComparer);

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

                if (!TryGetParameterValue(paramTuple, inputPositions, 1, out var target) ||
                    !IsConcretePairTarget(target))
                {
                    continue;
                }

                var pairKey = new object[groupCount + 2];
                var completeKey = true;
                for (var i = 0; i < groupCount; i++)
                {
                    if (!TryGetParameterValue(paramTuple, inputPositions, groupIndices[i], out var groupValue))
                    {
                        completeKey = false;
                        break;
                    }

                    pairKey[i] = groupValue!;
                }

                if (!completeKey)
                {
                    continue;
                }

                pairKey[groupCount] = source!;
                pairKey[groupCount + 1] = target!;
                seenPairs.Add(new RowWrapper(pairKey));
            }

            return Math.Max(1, seenPairs.Count);
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
                var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);
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
                var canReuseSeededCache = _seededCacheMaxEntries > 0;

                if (canReuseSeededCache &&
                    context.TransitiveClosureSeededResults.TryGetValue(cacheKey, out var cachedBySeed) &&
                    TryGetLruRowWrapperCacheValue(cachedBySeed, new RowWrapper(seedsKey), out var cachedRows))
                {
                    trace?.RecordCacheLookup("TransitiveClosureSeeded", traceKey, hit: true, built: false);
                    return cachedRows;
                }

                trace?.RecordCacheLookup("TransitiveClosureSeeded", traceKey, hit: false, built: true);

                const int MaxDagBitsetNodeCount = 16384;
                const int MinDagBitsetSeedCount = 128;
                const int MinDagBitsetEdgeCount = 8192;
                if (seeds.Count >= MinDagBitsetSeedCount &&
                    edges.Count >= MinDagBitsetEdgeCount &&
                    TryBuildDagReachabilityBitsets(edges, MaxDagBitsetNodeCount, out var dagNodeIds, out var dagNodesById, out var dagReachability))
                {
                    trace?.RecordStrategy(closure, "TransitiveClosureSeededDagBitset");
                    var dagRows = new List<object[]>();

                    foreach (var seed in seeds)
                    {
                        if (!dagNodeIds.TryGetValue(seed, out var seedId))
                        {
                            continue;
                        }

                        var reachable = dagReachability[seedId];
                        for (var targetId = 0; targetId < reachable.Length; targetId++)
                        {
                            if (!reachable[targetId])
                            {
                                continue;
                            }

                            dagRows.Add(new object[] { seed!, dagNodesById[targetId]! });
                        }
                    }

                    if (canReuseSeededCache)
                    {
                        if (!context.TransitiveClosureSeededResults.TryGetValue(cacheKey, out var dagStoreBySeed))
                        {
                            dagStoreBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                            context.TransitiveClosureSeededResults.Add(cacheKey, dagStoreBySeed);
                        }

                        var admitSeededCache = ShouldAdmitSeededCacheRows(
                            dagRows.Count,
                            seeds.Count,
                            useRowsPerSeedGate: true);
                        TryAdmitLruBoundedRowWrapperCacheEntry(
                            dagStoreBySeed,
                            new RowWrapper(seedsKey),
                            dagRows,
                            _seededCacheMaxEntries,
                            admitSeededCache,
                            trace,
                            "TransitiveClosureSeeded",
                            traceKey);
                    }

                    return dagRows;
                }

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

                    if (canReuseSeededCache)
                    {
                        if (!context.TransitiveClosureSeededResults.TryGetValue(cacheKey, out var memoizedStoreBySeed))
                        {
                            memoizedStoreBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                            context.TransitiveClosureSeededResults.Add(cacheKey, memoizedStoreBySeed);
                        }

                        var admitSeededCache = ShouldAdmitSeededCacheRows(
                            memoizedRows.Count,
                            seeds.Count,
                            useRowsPerSeedGate: true);
                        TryAdmitLruBoundedRowWrapperCacheEntry(
                            memoizedStoreBySeed,
                            new RowWrapper(seedsKey),
                            memoizedRows,
                            _seededCacheMaxEntries,
                            admitSeededCache,
                            trace,
                            "TransitiveClosureSeeded",
                            traceKey);
                    }

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

                    if (canReuseSeededCache)
                    {
                        if (!context.TransitiveClosureSeededResults.TryGetValue(cacheKey, out var singleStoreBySeed))
                        {
                            singleStoreBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                            context.TransitiveClosureSeededResults.Add(cacheKey, singleStoreBySeed);
                        }

                        var admitSeededCache = ShouldAdmitSeededCacheRows(
                            singleRows.Count,
                            seeds.Count,
                            useRowsPerSeedGate: true);
                        TryAdmitLruBoundedRowWrapperCacheEntry(
                            singleStoreBySeed,
                            new RowWrapper(seedsKey),
                            singleRows,
                            _seededCacheMaxEntries,
                            admitSeededCache,
                            trace,
                            "TransitiveClosureSeeded",
                            traceKey);
                    }

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

                if (canReuseSeededCache)
                {
                    if (!context.TransitiveClosureSeededResults.TryGetValue(cacheKey, out var storeBySeed))
                    {
                        storeBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                        context.TransitiveClosureSeededResults.Add(cacheKey, storeBySeed);
                    }

                    var admitSeededCache = ShouldAdmitSeededCacheRows(
                        totalRows.Count,
                        seeds.Count,
                        useRowsPerSeedGate: true);
                    TryAdmitLruBoundedRowWrapperCacheEntry(
                        storeBySeed,
                        new RowWrapper(seedsKey),
                        totalRows,
                        _seededCacheMaxEntries,
                        admitSeededCache,
                        trace,
                        "TransitiveClosureSeeded",
                        traceKey);
                }

                return totalRows;
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private static bool TryBuildDagReachabilityBitsets(
            IReadOnlyList<object[]> edges,
            int maxNodeCount,
            out Dictionary<object?, int> nodeIds,
            out object?[] nodesById,
            out BitArray[] reachability)
        {
            var localNodeIds = new Dictionary<object?, int>();
            nodeIds = localNodeIds;
            nodesById = Array.Empty<object?>();
            reachability = Array.Empty<BitArray>();

            if (edges.Count == 0)
            {
                return false;
            }

            var successors = new List<List<int>>();
            var indegree = new List<int>();
            var nodeValues = new List<object?>();

            int GetNodeId(object? value)
            {
                if (localNodeIds.TryGetValue(value, out var id))
                {
                    return id;
                }

                id = nodeValues.Count;
                localNodeIds.Add(value, id);
                nodeValues.Add(value);
                successors.Add(new List<int>());
                indegree.Add(0);
                return id;
            }

            foreach (var edge in edges)
            {
                if (edge is null || edge.Length < 2)
                {
                    continue;
                }

                var fromId = GetNodeId(edge[0]);
                var toId = GetNodeId(edge[1]);
                successors[fromId].Add(toId);
                indegree[toId]++;
            }

            if (nodeValues.Count == 0 || nodeValues.Count > maxNodeCount)
            {
                return false;
            }

            var queue = new Queue<int>();
            for (var i = 0; i < indegree.Count; i++)
            {
                if (indegree[i] == 0)
                {
                    queue.Enqueue(i);
                }
            }

            var topo = new List<int>(nodeValues.Count);
            while (queue.Count > 0)
            {
                var id = queue.Dequeue();
                topo.Add(id);
                foreach (var next in successors[id])
                {
                    indegree[next]--;
                    if (indegree[next] == 0)
                    {
                        queue.Enqueue(next);
                    }
                }
            }

            if (topo.Count != nodeValues.Count)
            {
                return false;
            }

            reachability = new BitArray[nodeValues.Count];
            for (var i = 0; i < reachability.Length; i++)
            {
                reachability[i] = new BitArray(nodeValues.Count);
            }

            for (var index = topo.Count - 1; index >= 0; index--)
            {
                var nodeId = topo[index];
                var bits = reachability[nodeId];
                foreach (var next in successors[nodeId])
                {
                    bits[next] = true;
                    bits.Or(reachability[next]);
                }
            }

            nodesById = nodeValues.ToArray();
            nodeIds = localNodeIds;
            return true;
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
                var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);
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
                var canReuseSeededCache = _seededCacheMaxEntries > 0;

                if (canReuseSeededCache &&
                    context.TransitiveClosureSeededByTargetResults.TryGetValue(cacheKey, out var cachedBySeed) &&
                    TryGetLruRowWrapperCacheValue(cachedBySeed, new RowWrapper(seedsKey), out var cachedRows))
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

                    if (canReuseSeededCache)
                    {
                        if (!context.TransitiveClosureSeededByTargetResults.TryGetValue(cacheKey, out var memoizedStoreBySeed))
                        {
                            memoizedStoreBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                            context.TransitiveClosureSeededByTargetResults.Add(cacheKey, memoizedStoreBySeed);
                        }

                        var admitSeededCache = ShouldAdmitSeededCacheRows(
                            memoizedRows.Count,
                            seeds.Count,
                            useRowsPerSeedGate: true);
                        TryAdmitLruBoundedRowWrapperCacheEntry(
                            memoizedStoreBySeed,
                            new RowWrapper(seedsKey),
                            memoizedRows,
                            _seededCacheMaxEntries,
                            admitSeededCache,
                            trace,
                            "TransitiveClosureSeededByTarget",
                            traceKey);
                    }

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

                    if (canReuseSeededCache)
                    {
                        if (!context.TransitiveClosureSeededByTargetResults.TryGetValue(cacheKey, out var singleStoreBySeed))
                        {
                            singleStoreBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                            context.TransitiveClosureSeededByTargetResults.Add(cacheKey, singleStoreBySeed);
                        }

                        var admitSeededCache = ShouldAdmitSeededCacheRows(
                            singleRows.Count,
                            seeds.Count,
                            useRowsPerSeedGate: true);
                        TryAdmitLruBoundedRowWrapperCacheEntry(
                            singleStoreBySeed,
                            new RowWrapper(seedsKey),
                            singleRows,
                            _seededCacheMaxEntries,
                            admitSeededCache,
                            trace,
                            "TransitiveClosureSeededByTarget",
                            traceKey);
                    }

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

                if (canReuseSeededCache)
                {
                    if (!context.TransitiveClosureSeededByTargetResults.TryGetValue(cacheKey, out var storeBySeed))
                    {
                        storeBySeed = new Dictionary<RowWrapper, IReadOnlyList<object[]>>(StructuralRowWrapperComparer);
                        context.TransitiveClosureSeededByTargetResults.Add(cacheKey, storeBySeed);
                    }

                    var admitSeededCache = ShouldAdmitSeededCacheRows(
                        totalRows.Count,
                        seeds.Count,
                        useRowsPerSeedGate: true);
                    TryAdmitLruBoundedRowWrapperCacheEntry(
                        storeBySeed,
                        new RowWrapper(seedsKey),
                        totalRows,
                        _seededCacheMaxEntries,
                        admitSeededCache,
                        trace,
                        "TransitiveClosureSeededByTarget",
                        traceKey);
                }

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
                var singleConcretePairRequest = IsSingleConcretePairRequest(bySource, byTarget);
                var preferForwardSingleProbe = true;
                Dictionary<object, List<object[]>>? singleSuccIndex = null;
                Dictionary<object, List<object[]>>? singlePredIndex = null;
                var singleForwardCost = 0;
                var singleBackwardCost = 0;

                if (singleConcretePairRequest)
                {
                    var source = bySource.First().Key;
                    var target = bySource.First().Value.First();
                    var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);
                    singleSuccIndex = GetFactIndex(closure.EdgeRelation, 0, edges, context);
                    singlePredIndex = GetFactIndex(closure.EdgeRelation, 1, edges, context);
                    singleForwardCost = CountEdgeBucket(singleSuccIndex, source);
                    singleBackwardCost = CountEdgeBucket(singlePredIndex, target);
                    preferForwardSingleProbe = singleForwardCost <= singleBackwardCost;
                }

                Dictionary<object?, HashSet<object?>> forwardBySource = new();
                Dictionary<object?, HashSet<object?>> backwardByTarget = new();
                var canBuildDirectionBatches = !singleConcretePairRequest &&
                    TryBuildPairProbeDirectionBatches(
                        closure,
                        bySource,
                        context,
                        out forwardBySource,
                        out backwardByTarget);
                var canReuseBatchedPairProbeCache =
                    canBuildDirectionBatches &&
                    _pairProbeCacheMaxEntries > 0 &&
                    _pairProbeCacheAdmissionMinCostPerProbe > 0d;
                var canMemoizeForwardBatch =
                    canBuildDirectionBatches &&
                    _cacheContext is not null &&
                    forwardBySource.Count > 0 &&
                    forwardBySource.Count <= MaxMemoizedPairSeeds;
                var canMemoizeBackwardBatch =
                    canBuildDirectionBatches &&
                    _cacheContext is not null &&
                    backwardByTarget.Count > 0 &&
                    backwardByTarget.Count <= MaxMemoizedPairSeeds;
                var canMemoizePairs =
                    !singleConcretePairRequest &&
                    _cacheContext is not null &&
                    bySource.Count <= MaxMemoizedPairSeeds &&
                    byTarget.Count <= MaxMemoizedPairSeeds;
                var preferForwardFallback = bySource.Count <= byTarget.Count;
                var pairRequestCount = CountClosurePairRequests(bySource.Values);
                IReadOnlyDictionary<ClosurePairPlanStrategy, Func<TimeSpan>>? measurePairStrategyProbes = null;
                if (_enableMeasuredClosurePairStrategy &&
                    !singleConcretePairRequest &&
                    pairRequestCount > 1 &&
                    pairRequestCount <= 16)
                {
                    var probes = new Dictionary<ClosurePairPlanStrategy, Func<TimeSpan>>
                    {
                        [ClosurePairPlanStrategy.Forward] = () => MeasureUngroupedClosurePairStrategyProbe(
                            closure,
                            ClosurePairPlanStrategy.Forward,
                            bySource,
                            byTarget,
                            forwardBySource,
                            backwardByTarget,
                            context),
                        [ClosurePairPlanStrategy.Backward] = () => MeasureUngroupedClosurePairStrategyProbe(
                            closure,
                            ClosurePairPlanStrategy.Backward,
                            bySource,
                            byTarget,
                            forwardBySource,
                            backwardByTarget,
                            context)
                    };

                    if (canMemoizeForwardBatch || canMemoizePairs)
                    {
                        probes[ClosurePairPlanStrategy.MemoizedBySource] = () => MeasureUngroupedClosurePairStrategyProbe(
                            closure,
                            ClosurePairPlanStrategy.MemoizedBySource,
                            bySource,
                            byTarget,
                            forwardBySource,
                            backwardByTarget,
                            context);
                    }

                    if (canMemoizeBackwardBatch || canMemoizePairs)
                    {
                        probes[ClosurePairPlanStrategy.MemoizedByTarget] = () => MeasureUngroupedClosurePairStrategyProbe(
                            closure,
                            ClosurePairPlanStrategy.MemoizedByTarget,
                            bySource,
                            byTarget,
                            forwardBySource,
                            backwardByTarget,
                            context);
                    }

                    if (canBuildDirectionBatches &&
                        forwardBySource.Count > 0 &&
                        backwardByTarget.Count > 0)
                    {
                        probes[ClosurePairPlanStrategy.MixedDirection] = () => MeasureUngroupedClosurePairStrategyProbe(
                            closure,
                            ClosurePairPlanStrategy.MixedDirection,
                            bySource,
                            byTarget,
                            forwardBySource,
                            backwardByTarget,
                            context);

                        if (canReuseBatchedPairProbeCache)
                        {
                            probes[ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache] = () => MeasureUngroupedClosurePairStrategyProbe(
                                closure,
                                ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache,
                                bySource,
                                byTarget,
                                forwardBySource,
                                backwardByTarget,
                                context);
                        }
                    }

                    measurePairStrategyProbes = probes;
                }

                var pairStrategySelection = ResolveClosurePairStrategy(
                    trace,
                    closure,
                    _closurePairStrategy,
                    pairRequestCount,
                    bySource.Count,
                    byTarget.Count,
                    singleConcretePairRequest,
                    preferForwardSingleProbe,
                    canBuildDirectionBatches,
                    forwardBySource.Count > 0,
                    backwardByTarget.Count > 0,
                    canReuseBatchedPairProbeCache,
                    canMemoizeForwardBatch,
                    canMemoizeBackwardBatch,
                    canMemoizePairs,
                    preferForwardFallback,
                    _useSeededClosureCachesForPairBatches && _cacheContext is not null,
                    measurePairStrategyProbes);
                RecordClosurePairStrategy(trace, closure, "TransitiveClosurePairs", pairStrategySelection);

                return pairStrategySelection.Strategy switch
                {
                    ClosurePairPlanStrategy.SingleProbeForward => ExecuteSeededTransitiveClosurePairsSingleProbe(
                        closure,
                        parameters,
                        bySource,
                        byTarget,
                        context,
                        preferForward: true,
                        singleSuccIndex!,
                        singlePredIndex!,
                        singleForwardCost,
                        singleBackwardCost),
                    ClosurePairPlanStrategy.SingleProbeBackward => ExecuteSeededTransitiveClosurePairsSingleProbe(
                        closure,
                        parameters,
                        bySource,
                        byTarget,
                        context,
                        preferForward: false,
                        singleSuccIndex!,
                        singlePredIndex!,
                        singleForwardCost,
                        singleBackwardCost),
                    ClosurePairPlanStrategy.MixedDirectionWithPairProbeCache => ExecuteSeededTransitiveClosurePairsMixedDirectionWithPairProbeCache(
                        closure,
                        forwardBySource,
                        backwardByTarget,
                        context),
                    ClosurePairPlanStrategy.MixedDirection => ExecuteSeededTransitiveClosurePairsMixedDirection(
                        closure,
                        forwardBySource,
                        backwardByTarget,
                        context),
                    ClosurePairPlanStrategy.MemoizedBySource => ExecuteSeededTransitiveClosurePairsMemoizedBySource(
                        closure,
                        bySource,
                        context),
                    ClosurePairPlanStrategy.MemoizedByTarget => ExecuteSeededTransitiveClosurePairsMemoizedByTarget(
                        closure,
                        byTarget,
                        context),
                    ClosurePairPlanStrategy.Forward => ExecuteSeededTransitiveClosurePairsForward(
                        closure,
                        bySource,
                        context),
                    _ => ExecuteSeededTransitiveClosurePairsBackward(
                        closure,
                        byTarget,
                        context)
                };
            }
            finally
            {
                context.FixpointDepth--;
            }
        }

        private IEnumerable<object[]> ExecuteSeededTransitiveClosurePairsSingleProbe(
            TransitiveClosureNode closure,
            IReadOnlyList<object[]> parameters,
            IReadOnlyDictionary<object?, HashSet<object?>> bySource,
            IReadOnlyDictionary<object?, HashSet<object?>> byTarget,
            EvaluationContext context,
            bool preferForward,
            Dictionary<object, List<object[]>> succIndex,
            Dictionary<object, List<object[]>> predIndex,
            int forwardCost,
            int backwardCost)
        {
            var trace = context.Trace;
            var probeCount = ComputeSinglePairProbeNormalizationCount(parameters);
            var source = bySource.First().Key;
            var target = bySource.First().Value.First();
            var pairCacheKey = (closure.EdgeRelation, closure.Predicate);
            var pairProbeKey = new object[] { source!, target! };
            var sourceLabel = source is null ? "<null>" : FormatCacheSeedValue(source);
            var targetLabel = target is null ? "<null>" : FormatCacheSeedValue(target);
            var traceKey =
                $"{closure.Predicate.Name}/{closure.Predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:pair={sourceLabel}->{targetLabel}";
            var canReusePairProbeCache = _pairProbeCacheMaxEntries > 0;

            if (canReusePairProbeCache &&
                context.TransitiveClosurePairProbeResults.TryGetValue(pairCacheKey, out var cachedByPair) &&
                TryGetLruRowWrapperCacheValue(cachedByPair, new RowWrapper(pairProbeKey), out var pairReachable))
            {
                trace?.RecordCacheLookup("TransitiveClosurePairsSingleProbe", traceKey, hit: true, built: false);
                if (!pairReachable)
                {
                    return Array.Empty<object[]>();
                }

                return new List<object[]> { new object[] { source!, target! } };
            }

            trace?.RecordCacheLookup("TransitiveClosurePairsSingleProbe", traceKey, hit: false, built: true);

            var rows = preferForward
                ? ExecuteSeededTransitiveClosurePairsForward(closure, bySource, context).ToList()
                : ExecuteSeededTransitiveClosurePairsBackward(closure, byTarget, context).ToList();

            if (canReusePairProbeCache)
            {
                if (!context.TransitiveClosurePairProbeResults.TryGetValue(pairCacheKey, out var pairStore))
                {
                    pairStore = new Dictionary<RowWrapper, bool>(StructuralRowWrapperComparer);
                    context.TransitiveClosurePairProbeResults.Add(pairCacheKey, pairStore);
                }

                var admitPairProbeCache = ShouldAdmitPairProbeCacheEntry(
                    forwardCost,
                    backwardCost,
                    probeCount,
                    useCostPerProbeGate: true);
                TryAdmitLruBoundedRowWrapperCacheEntry(
                    pairStore,
                    new RowWrapper(pairProbeKey),
                    rows.Count > 0,
                    _pairProbeCacheMaxEntries,
                    admitPairProbeCache,
                    trace,
                    "TransitiveClosurePairsSingleProbe",
                    traceKey);
            }

            return rows;
        }

        private IEnumerable<object[]> ExecuteSeededTransitiveClosurePairsMixedDirection(
            TransitiveClosureNode closure,
            IReadOnlyDictionary<object?, HashSet<object?>> forwardBySource,
            IReadOnlyDictionary<object?, HashSet<object?>> backwardByTarget,
            EvaluationContext context)
        {
            if (_useSeededClosureCachesForPairBatches && _cacheContext is not null)
            {
                foreach (var row in ExecuteSeededTransitiveClosurePairsMemoizedBySource(closure, forwardBySource, context))
                {
                    yield return row;
                }

                foreach (var row in ExecuteSeededTransitiveClosurePairsMemoizedByTarget(closure, backwardByTarget, context))
                {
                    yield return row;
                }

                yield break;
            }

            foreach (var row in ExecuteSeededTransitiveClosurePairsForward(closure, forwardBySource, context))
            {
                yield return row;
            }

            foreach (var row in ExecuteSeededTransitiveClosurePairsBackward(closure, backwardByTarget, context))
            {
                yield return row;
            }
        }

        private IEnumerable<object[]> ExecuteSeededTransitiveClosurePairsMixedDirectionWithPairProbeCache(
            TransitiveClosureNode closure,
            IReadOnlyDictionary<object?, HashSet<object?>> forwardBySource,
            IReadOnlyDictionary<object?, HashSet<object?>> backwardByTarget,
            EvaluationContext context)
        {
            var trace = context.Trace;
            var pairCacheKey = (closure.EdgeRelation, closure.Predicate);
            if (!context.TransitiveClosurePairProbeResults.TryGetValue(pairCacheKey, out var pairStore))
            {
                pairStore = new Dictionary<RowWrapper, bool>(StructuralRowWrapperComparer);
                context.TransitiveClosurePairProbeResults.Add(pairCacheKey, pairStore);
            }

            var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);
            var succIndex = GetFactIndex(closure.EdgeRelation, 0, edges, context);
            var predIndex = GetFactIndex(closure.EdgeRelation, 1, edges, context);
            var sourceCostCache = new Dictionary<object?, int>();
            var targetCostCache = new Dictionary<object?, int>();

            var cachedRows = new List<object[]>();
            var missedPairs = new List<(object? Source, object? Target, int ForwardCost, int BackwardCost)>();
            var forwardMisses = new Dictionary<object?, HashSet<object?>>();
            var backwardMisses = new Dictionary<object?, HashSet<object?>>();

            foreach (var sourceEntry in forwardBySource)
            {
                var source = sourceEntry.Key;
                if (!sourceCostCache.TryGetValue(source, out var forwardCost))
                {
                    forwardCost = CountEdgeBucket(succIndex, source);
                    sourceCostCache.Add(source, forwardCost);
                }

                foreach (var target in sourceEntry.Value)
                {
                    if (!targetCostCache.TryGetValue(target, out var backwardCost))
                    {
                        backwardCost = CountEdgeBucket(predIndex, target);
                        targetCostCache.Add(target, backwardCost);
                    }

                    var pairProbeKey = new object[] { source!, target! };
                    var sourceLabel = source is null ? "<null>" : FormatCacheSeedValue(source);
                    var targetLabel = target is null ? "<null>" : FormatCacheSeedValue(target);
                    var traceKey =
                        $"{closure.Predicate.Name}/{closure.Predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:pair={sourceLabel}->{targetLabel}";

                    if (TryGetLruRowWrapperCacheValue(pairStore, new RowWrapper(pairProbeKey), out var pairReachable))
                    {
                        trace?.RecordCacheLookup("TransitiveClosurePairsSingleProbe", traceKey, hit: true, built: false);
                        if (pairReachable)
                        {
                            cachedRows.Add(new object[] { source!, target! });
                        }
                    }
                    else
                    {
                        trace?.RecordCacheLookup("TransitiveClosurePairsSingleProbe", traceKey, hit: false, built: true);
                        AddPairRequest(forwardMisses, source, target);
                        missedPairs.Add((source, target, forwardCost, backwardCost));
                    }
                }
            }

            foreach (var targetEntry in backwardByTarget)
            {
                var target = targetEntry.Key;
                if (!targetCostCache.TryGetValue(target, out var backwardCost))
                {
                    backwardCost = CountEdgeBucket(predIndex, target);
                    targetCostCache.Add(target, backwardCost);
                }

                foreach (var source in targetEntry.Value)
                {
                    if (!sourceCostCache.TryGetValue(source, out var forwardCost))
                    {
                        forwardCost = CountEdgeBucket(succIndex, source);
                        sourceCostCache.Add(source, forwardCost);
                    }

                    var pairProbeKey = new object[] { source!, target! };
                    var sourceLabel = source is null ? "<null>" : FormatCacheSeedValue(source);
                    var targetLabel = target is null ? "<null>" : FormatCacheSeedValue(target);
                    var traceKey =
                        $"{closure.Predicate.Name}/{closure.Predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:pair={sourceLabel}->{targetLabel}";

                    if (TryGetLruRowWrapperCacheValue(pairStore, new RowWrapper(pairProbeKey), out var pairReachable))
                    {
                        trace?.RecordCacheLookup("TransitiveClosurePairsSingleProbe", traceKey, hit: true, built: false);
                        if (pairReachable)
                        {
                            cachedRows.Add(new object[] { source!, target! });
                        }
                    }
                    else
                    {
                        trace?.RecordCacheLookup("TransitiveClosurePairsSingleProbe", traceKey, hit: false, built: true);
                        AddPairRequest(backwardMisses, target, source);
                        missedPairs.Add((source, target, forwardCost, backwardCost));
                    }
                }
            }

            List<object[]> computedRows;
            if (forwardMisses.Count > 0 && backwardMisses.Count > 0)
            {
                computedRows = ExecuteSeededTransitiveClosurePairsMixedDirection(
                    closure,
                    forwardMisses,
                    backwardMisses,
                    context).ToList();
            }
            else if (forwardMisses.Count > 0)
            {
                computedRows = ExecuteSeededTransitiveClosurePairsForward(
                    closure,
                    forwardMisses,
                    context).ToList();
            }
            else if (backwardMisses.Count > 0)
            {
                computedRows = ExecuteSeededTransitiveClosurePairsBackward(
                    closure,
                    backwardMisses,
                    context).ToList();
            }
            else
            {
                computedRows = new List<object[]>();
            }

            var reachablePairs = new HashSet<RowWrapper>(StructuralRowWrapperComparer);
            foreach (var row in computedRows)
            {
                if (row is null || row.Length < 2)
                {
                    continue;
                }

                reachablePairs.Add(new RowWrapper(new object[] { row[0], row[1] }));
            }

            var probeCount = ComputeMixedPairProbeNormalizationCount(missedPairs);
            foreach (var pair in missedPairs)
            {
                var pairProbeKey = new object[] { pair.Source!, pair.Target! };
                var sourceLabel = pair.Source is null ? "<null>" : FormatCacheSeedValue(pair.Source);
                var targetLabel = pair.Target is null ? "<null>" : FormatCacheSeedValue(pair.Target);
                var traceKey =
                    $"{closure.Predicate.Name}/{closure.Predicate.Arity}:edge={closure.EdgeRelation.Name}/{closure.EdgeRelation.Arity}:pair={sourceLabel}->{targetLabel}";
                var admitPairProbeCache = ShouldAdmitPairProbeCacheEntry(
                    pair.ForwardCost,
                    pair.BackwardCost,
                    probeCount,
                    useCostPerProbeGate: true);
                var pairReachable = reachablePairs.Contains(new RowWrapper(pairProbeKey));

                TryAdmitLruBoundedRowWrapperCacheEntry(
                    pairStore,
                    new RowWrapper(pairProbeKey),
                    pairReachable,
                    _pairProbeCacheMaxEntries,
                    admitPairProbeCache,
                    trace,
                    "TransitiveClosurePairsSingleProbe",
                    traceKey);
            }

            if (cachedRows.Count == 0)
            {
                return computedRows;
            }

            if (computedRows.Count == 0)
            {
                return cachedRows;
            }

            cachedRows.AddRange(computedRows);
            return cachedRows;
        }

        private bool TryBuildPairProbeDirectionBatches(
            TransitiveClosureNode closure,
            IReadOnlyDictionary<object?, HashSet<object?>> bySource,
            EvaluationContext context,
            out Dictionary<object?, HashSet<object?>> forwardBySource,
            out Dictionary<object?, HashSet<object?>> backwardByTarget)
        {
            forwardBySource = new Dictionary<object?, HashSet<object?>>();
            backwardByTarget = new Dictionary<object?, HashSet<object?>>();

            if (!IsMultiConcretePairRequest(bySource))
            {
                return false;
            }

            var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);
            var succIndex = GetFactIndex(closure.EdgeRelation, 0, edges, context);
            var predIndex = GetFactIndex(closure.EdgeRelation, 1, edges, context);
            var sourceCostCache = new Dictionary<object?, int>();
            var targetCostCache = new Dictionary<object?, int>();

            foreach (var sourceEntry in bySource)
            {
                var source = sourceEntry.Key;
                if (!sourceCostCache.TryGetValue(source, out var forwardCost))
                {
                    forwardCost = CountEdgeBucket(succIndex, source);
                    sourceCostCache.Add(source, forwardCost);
                }

                foreach (var target in sourceEntry.Value)
                {
                    if (!IsConcretePairTarget(target))
                    {
                        forwardBySource.Clear();
                        backwardByTarget.Clear();
                        return false;
                    }

                    if (!targetCostCache.TryGetValue(target, out var backwardCost))
                    {
                        backwardCost = CountEdgeBucket(predIndex, target);
                        targetCostCache.Add(target, backwardCost);
                    }

                    if (forwardCost <= backwardCost)
                    {
                        AddPairRequest(forwardBySource, source, target);
                    }
                    else
                    {
                        AddPairRequest(backwardByTarget, target, source);
                    }
                }
            }

            return forwardBySource.Count > 0 || backwardByTarget.Count > 0;
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
            var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);
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
            var edges = GetClosureFactsList(closure.EdgeRelation, ClosureRelationAccessKind.Edge, context, closure);
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
                    // Max iteration bound: prevents non-convergence on cyclic
                    // graphs with counter-bearing recursive predicates.
                    // With d_eff formula (n=5), iterations beyond 50 contribute
                    // negligibly: d^(-5) at depth 50 ≈ 3e-9.
                    if (_maxFixpointIterations > 0 && iteration > _maxFixpointIterations)
                    {
                        break;
                    }
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
                ReplayableFactSources = parent?.ReplayableFactSources ?? new Dictionary<PredicateId, IReplayableRelationSource>();
                ScanRelationRetentionSelections = parent?.ScanRelationRetentionSelections ?? new Dictionary<(PredicateId Predicate, ScanRelationAccessKind AccessKind), ScanRelationRetentionSelection>();
                ClosureRelationRetentionSelections = parent?.ClosureRelationRetentionSelections ?? new Dictionary<(PredicateId Predicate, ClosureRelationAccessKind AccessKind), ClosureRelationRetentionSelection>();
                PathAwareSupportRelationRetentionSelections = parent?.PathAwareSupportRelationRetentionSelections ?? new Dictionary<(PredicateId Predicate, PathAwareSupportRelationAccessKind AccessKind), PathAwareSupportRelationRetentionSelection>();
                PathAwareEdgeStates = parent?.PathAwareEdgeStates ?? new Dictionary<PredicateId, PathAwareEdgeState>();
                PathAwareEdgeRetentionSelections = parent?.PathAwareEdgeRetentionSelections ?? new Dictionary<PredicateId, PathAwareEdgeRetentionSelection>();
                FactSets = parent?.FactSets ?? new Dictionary<PredicateId, HashSet<object[]>>();
                FactIndices = parent?.FactIndices ?? new Dictionary<(PredicateId Predicate, int ColumnIndex), Dictionary<object, List<object[]>>>();
                JoinIndices = parent?.JoinIndices ?? new Dictionary<(PredicateId Predicate, string KeySignature), Dictionary<RowWrapper, List<object[]>>>();
                TransitiveClosureResults = parent?.TransitiveClosureResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId Predicate), IReadOnlyList<object[]>>();
                TransitiveClosureSeededResults = parent?.TransitiveClosureSeededResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId Predicate), Dictionary<RowWrapper, IReadOnlyList<object[]>>>();
                TransitiveClosureSeededByTargetResults = parent?.TransitiveClosureSeededByTargetResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId Predicate), Dictionary<RowWrapper, IReadOnlyList<object[]>>>();
                TransitiveClosurePairProbeResults = parent?.TransitiveClosurePairProbeResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId Predicate), Dictionary<RowWrapper, bool>>();
                GroupedTransitiveClosureResults = parent?.GroupedTransitiveClosureResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId Predicate, string Groups), IReadOnlyList<object[]>>();
                SeedGroupedTransitiveClosureCountResults = parent?.SeedGroupedTransitiveClosureCountResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId SeedRelation, PredicateId Predicate), IReadOnlyList<object[]>>();
                SeedGroupedDagLongestDepthResults = parent?.SeedGroupedDagLongestDepthResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId SeedRelation, PredicateId Predicate), IReadOnlyList<object[]>>();
                SeedGroupedPathAwareWeightSumResults = parent?.SeedGroupedPathAwareWeightSumResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId SeedRelation, PredicateId RootRelation, PredicateId Predicate, double DistanceExponent, int MaxDepth), IReadOnlyList<object[]>>();
                SeedGroupedPathAwareDepthMinResults = parent?.SeedGroupedPathAwareDepthMinResults
                    ?? new Dictionary<SeedGroupedPathAwareDepthMinNode, IReadOnlyList<object[]>>();
                SeedGroupedPathAwareAccumulationMinResults = parent?.SeedGroupedPathAwareAccumulationMinResults
                    ?? new Dictionary<SeedGroupedPathAwareAccumulationMinNode, IReadOnlyList<object[]>>();
                GroupedTransitiveClosureSeededResults = parent?.GroupedTransitiveClosureSeededResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId Predicate, string Groups), Dictionary<RowWrapper, IReadOnlyList<object[]>>>();
                GroupedTransitiveClosureSeededByTargetResults = parent?.GroupedTransitiveClosureSeededByTargetResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId Predicate, string Groups), Dictionary<RowWrapper, IReadOnlyList<object[]>>>();
                GroupedTransitiveClosurePairProbeResults = parent?.GroupedTransitiveClosurePairProbeResults
                    ?? new Dictionary<(PredicateId EdgeRelation, PredicateId Predicate, string Groups), Dictionary<RowWrapper, bool>>();
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

            public Dictionary<PredicateId, IReplayableRelationSource> ReplayableFactSources { get; }

            public Dictionary<(PredicateId Predicate, ScanRelationAccessKind AccessKind), ScanRelationRetentionSelection> ScanRelationRetentionSelections { get; }

            public Dictionary<(PredicateId Predicate, ClosureRelationAccessKind AccessKind), ClosureRelationRetentionSelection> ClosureRelationRetentionSelections { get; }

            public Dictionary<(PredicateId Predicate, PathAwareSupportRelationAccessKind AccessKind), PathAwareSupportRelationRetentionSelection> PathAwareSupportRelationRetentionSelections { get; }

            public Dictionary<PredicateId, PathAwareEdgeState> PathAwareEdgeStates { get; }

            public Dictionary<PredicateId, PathAwareEdgeRetentionSelection> PathAwareEdgeRetentionSelections { get; }

            public Dictionary<PredicateId, HashSet<object[]>> FactSets { get; }

            public Dictionary<(PredicateId Predicate, int ColumnIndex), Dictionary<object, List<object[]>>> FactIndices { get; }

            public Dictionary<(PredicateId Predicate, string KeySignature), Dictionary<RowWrapper, List<object[]>>> JoinIndices { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId Predicate), IReadOnlyList<object[]>> TransitiveClosureResults { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId Predicate), Dictionary<RowWrapper, IReadOnlyList<object[]>>> TransitiveClosureSeededResults { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId Predicate), Dictionary<RowWrapper, IReadOnlyList<object[]>>> TransitiveClosureSeededByTargetResults { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId Predicate), Dictionary<RowWrapper, bool>> TransitiveClosurePairProbeResults { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId Predicate, string Groups), IReadOnlyList<object[]>> GroupedTransitiveClosureResults { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId SeedRelation, PredicateId Predicate), IReadOnlyList<object[]>> SeedGroupedTransitiveClosureCountResults { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId SeedRelation, PredicateId Predicate), IReadOnlyList<object[]>> SeedGroupedDagLongestDepthResults { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId SeedRelation, PredicateId RootRelation, PredicateId Predicate, double DistanceExponent, int MaxDepth), IReadOnlyList<object[]>> SeedGroupedPathAwareWeightSumResults { get; }

            public Dictionary<SeedGroupedPathAwareDepthMinNode, IReadOnlyList<object[]>> SeedGroupedPathAwareDepthMinResults { get; }

            public Dictionary<SeedGroupedPathAwareAccumulationMinNode, IReadOnlyList<object[]>> SeedGroupedPathAwareAccumulationMinResults { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId Predicate, string Groups), Dictionary<RowWrapper, IReadOnlyList<object[]>>> GroupedTransitiveClosureSeededResults { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId Predicate, string Groups), Dictionary<RowWrapper, IReadOnlyList<object[]>>> GroupedTransitiveClosureSeededByTargetResults { get; }

            public Dictionary<(PredicateId EdgeRelation, PredicateId Predicate, string Groups), Dictionary<RowWrapper, bool>> GroupedTransitiveClosurePairProbeResults { get; }
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
