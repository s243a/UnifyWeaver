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
    /// Scans a base relation provided by <see cref="IRelationProvider"/>.
    /// </summary>
    public sealed record RelationScanNode(PredicateId Relation) : PlanNode;

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

        private sealed class ReferencePlanNodeComparer : IEqualityComparer<PlanNode>
        {
            public static readonly ReferencePlanNodeComparer Instance = new();

            public bool Equals(PlanNode? x, PlanNode? y) => ReferenceEquals(x, y);

            public int GetHashCode(PlanNode obj) => RuntimeHelpers.GetHashCode(obj);
        }

        private readonly Dictionary<PlanNode, NodeStats> _stats = new(ReferencePlanNodeComparer.Instance);
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

        internal void RecordInvocation(PlanNode node)
        {
            if (node is null) throw new ArgumentNullException(nameof(node));
            GetOrAdd(node).Invocations++;
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

        public override string ToString()
        {
            var lines = Snapshot()
                .Select(s =>
                    $"[{s.Id}] {s.NodeType} invocations={s.Invocations} enumerations={s.Enumerations} rows={s.Rows} elapsed={s.Elapsed}");
            return string.Join(Environment.NewLine, lines);
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
                    case RecursiveRefNode:
                    case CrossRefNode:
                    case EmptyNode:
                    case UnitNode:
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
                RelationScanNode scan => $"RelationScan relation={scan.Relation}",
                SelectionNode => "Selection",
                NegationNode negation => $"Negation predicate={negation.Predicate}",
                AggregateNode aggregate => $"Aggregate predicate={aggregate.Predicate} op={aggregate.Operation} groupBy=[{string.Join(",", aggregate.GroupByIndices)}] valueIndex={aggregate.ValueIndex} width={aggregate.Width}",
                AggregateSubplanNode aggregateSubplan => $"AggregateSubplan op={aggregateSubplan.Operation} groupBy=[{string.Join(",", aggregateSubplan.GroupByIndices)}] valueIndex={aggregateSubplan.ValueIndex} width={aggregateSubplan.Width}",
                ProjectionNode => "Projection",
                KeyJoinNode join => $"KeyJoin leftKeys=[{string.Join(",", join.LeftKeys)}] rightKeys=[{string.Join(",", join.RightKeys)}] width={join.Width}",
                JoinNode => "Join",
                UnionNode union => $"Union sources={union.Sources.Count}",
                DistinctNode distinct => $"Distinct comparer={(distinct.Comparer?.GetType().Name ?? "default")}",
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
        private readonly EvaluationContext? _cacheContext;

        public QueryExecutor(IRelationProvider provider, QueryExecutorOptions? options = null)
        {
            _provider = provider ?? throw new ArgumentNullException(nameof(provider));
            options ??= new QueryExecutorOptions();
            _cacheContext = options.ReuseCaches ? new EvaluationContext() : null;
        }

        public IEnumerable<object[]> Execute(QueryPlan plan, IEnumerable<object[]>? parameters = null, QueryExecutionTrace? trace = null)
        {
            if (plan is null) throw new ArgumentNullException(nameof(plan));
            var paramList = parameters?.ToList() ?? new List<object[]>();
            var context = _cacheContext is null
                ? new EvaluationContext(paramList, trace: trace)
                : new EvaluationContext(paramList, parent: _cacheContext, trace: trace);
            var inputPositions = plan.InputPositions;
            if (inputPositions is { Count: > 0 })
            {
                if (paramList.Count == 0)
                {
                    return Enumerable.Empty<object[]>();
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

        public void ClearCaches()
        {
            if (_cacheContext is null)
            {
                return;
            }

            _cacheContext.Facts.Clear();
            _cacheContext.FactSets.Clear();
            _cacheContext.FactIndices.Clear();
            _cacheContext.JoinIndices.Clear();
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

                case RelationScanNode scan:
                    result = context is null
                        ? _provider.GetFacts(scan.Relation) ?? Enumerable.Empty<object[]>()
                        : GetFactsList(scan.Relation, context);
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

                case FixpointNode fixpoint:
                    result = ExecuteFixpoint(fixpoint, context);
                    break;

                case MutualFixpointNode mutualFixpoint:
                    result = ExecuteMutualFixpoint(mutualFixpoint, context);
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

            return trace is null ? result : trace.WrapEnumeration(node, result);
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

        private IEnumerable<object[]> ExecuteKeyJoin(KeyJoinNode join, EvaluationContext? context)
        {
            if (join.LeftKeys is null || join.RightKeys is null || join.LeftKeys.Count == 0 || join.RightKeys.Count == 0)
            {
                var leftRows = Evaluate(join.Left, context);
                var rightRows = Evaluate(join.Right, context).ToList();
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

            if (context is not null && (join.Left is RelationScanNode || join.Right is RelationScanNode))
            {
                if (join.Left is RelationScanNode leftScan && join.Right is RelationScanNode rightScan)
                {
                    var leftFacts = GetFactsList(leftScan.Relation, context);
                    var rightFacts = GetFactsList(rightScan.Relation, context);
                    var keyCount = join.LeftKeys.Count;

                    if (leftFacts.Count <= rightFacts.Count)
                    {
                        var probe = Evaluate(join.Right, context);

                        if (keyCount == 1)
                        {
                            var index = GetFactIndex(leftScan.Relation, join.LeftKeys[0], leftFacts, context);
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
                        }
                        else
                        {
                            var index = GetJoinIndex(leftScan.Relation, join.LeftKeys, leftFacts, context);
                            foreach (var rightTuple in probe)
                            {
                                if (rightTuple is null) continue;

                                var key = BuildKeyFromTuple(rightTuple, join.RightKeys);
                                var wrapper = new RowWrapper(key);

                                if (!index.TryGetValue(wrapper, out var bucket))
                                {
                                    continue;
                                }

                                foreach (var leftTuple in bucket)
                                {
                                    yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                }
                            }
                        }
                    }
                    else
                    {
                        var probe = Evaluate(join.Left, context);

                        if (keyCount == 1)
                        {
                            var index = GetFactIndex(rightScan.Relation, join.RightKeys[0], rightFacts, context);
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
                        }
                        else
                        {
                            var index = GetJoinIndex(rightScan.Relation, join.RightKeys, rightFacts, context);
                            foreach (var leftTuple in probe)
                            {
                                if (leftTuple is null) continue;

                                var key = BuildKeyFromTuple(leftTuple, join.LeftKeys);
                                var wrapper = new RowWrapper(key);

                                if (!index.TryGetValue(wrapper, out var bucket))
                                {
                                    continue;
                                }

                                foreach (var rightTuple in bucket)
                                {
                                    yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                                }
                            }
                        }
                    }

                    yield break;
                }

                if (join.Right is RelationScanNode scan)
                {
                    var facts = GetFactsList(scan.Relation, context);
                    var probe = Evaluate(join.Left, context);
                    var keyCount = join.RightKeys.Count;

                    if (keyCount == 1)
                    {
                        var index = GetFactIndex(scan.Relation, join.RightKeys[0], facts, context);
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
                    }
                    else
                    {
                        var index = GetJoinIndex(scan.Relation, join.RightKeys, facts, context);
                        foreach (var leftTuple in probe)
                        {
                            if (leftTuple is null) continue;

                            var key = BuildKeyFromTuple(leftTuple, join.LeftKeys);
                            var wrapper = new RowWrapper(key);

                            if (!index.TryGetValue(wrapper, out var bucket))
                            {
                                continue;
                            }

                            foreach (var rightTuple in bucket)
                            {
                                yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                            }
                        }
                    }

                    yield break;
                }

                if (join.Left is RelationScanNode leftOnlyScan)
                {
                    var facts = GetFactsList(leftOnlyScan.Relation, context);
                    var probe = Evaluate(join.Right, context);
                    var keyCount = join.LeftKeys.Count;

                    if (keyCount == 1)
                    {
                        var index = GetFactIndex(leftOnlyScan.Relation, join.LeftKeys[0], facts, context);
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
                    }
                    else
                    {
                        var index = GetJoinIndex(leftOnlyScan.Relation, join.LeftKeys, facts, context);
                        foreach (var rightTuple in probe)
                        {
                            if (rightTuple is null) continue;

                            var key = BuildKeyFromTuple(rightTuple, join.RightKeys);
                            var wrapper = new RowWrapper(key);

                            if (!index.TryGetValue(wrapper, out var bucket))
                            {
                                continue;
                            }

                            foreach (var leftTuple in bucket)
                            {
                                yield return BuildJoinOutput(leftTuple, rightTuple, join.LeftWidth, join.RightWidth, join.Width);
                            }
                        }
                    }

                    yield break;
                }
            }

            var left = Evaluate(join.Left, context);
            var right = Evaluate(join.Right, context);
            var indexFallback = new Dictionary<RowWrapper, List<object[]>>(new RowWrapperComparer(StructuralArrayComparer.Instance));

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
                return cached;
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
                return cached;
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
                return cached;
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

        private Dictionary<object, List<object[]>> GetFactIndex(
            PredicateId predicate,
            int columnIndex,
            IReadOnlyList<object[]> facts,
            EvaluationContext context)
        {
            var cacheKey = (predicate, columnIndex);
            if (context.FactIndices.TryGetValue(cacheKey, out var cached))
            {
                return cached;
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

            List<object[]>? bestBucket = null;
            var bestCount = int.MaxValue;

            for (var i = 0; i < pattern.Length; i++)
            {
                var value = pattern[i];
                if (ReferenceEquals(value, Wildcard.Value))
                {
                    continue;
                }

                var index = GetFactIndex(predicate, i, facts, context);
                var key = value ?? NullFactIndexKey;

                if (!index.TryGetValue(key, out var bucket))
                {
                    return Array.Empty<object[]>();
                }

                if (bucket.Count < bestCount)
                {
                    bestCount = bucket.Count;
                    bestBucket = bucket;

                    if (bestCount <= 1)
                    {
                        break;
                    }
                }
            }

            if (bestBucket is not null)
            {
                return bestBucket;
            }

            return facts;
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

            var parameterSet = new HashSet<RowWrapper>(
                parameters.Select(p => new RowWrapper(BuildKeyFromTuple(p, inputPositions))),
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
                    var keyTuple = BuildKeyFromTuple(paramTuple, inputPositions);
                    keys.Add(keyTuple.Length > 0 ? keyTuple[0] ?? NullFactIndexKey : NullFactIndexKey);
                }

                return EnumerateBuckets(index, keys);
            }

            var joinIndex = GetJoinIndex(scan.Relation, inputPositions, facts, context);
            var keySet = new HashSet<RowWrapper>(new RowWrapperComparer(StructuralArrayComparer.Instance));

            foreach (var paramTuple in parameters)
            {
                if (paramTuple is null) continue;
                var key = BuildKeyFromTuple(paramTuple, inputPositions);
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
                for (var i = 0; i < inputPositions.Count; i++)
                {
                    key[i] = tuple[i];
                }
                return key;
            }

            for (var i = 0; i < inputPositions.Count; i++)
            {
                var pos = inputPositions[i];
                key[i] = pos >= 0 && pos < tuple.Length ? tuple[pos] : null!;
            }

            return key;
        }

        private IEnumerable<object[]> ExecuteFixpoint(FixpointNode fixpoint, EvaluationContext? parentContext)
        {
            if (fixpoint is null) throw new ArgumentNullException(nameof(fixpoint));

            var comparer = StructuralArrayComparer.Instance;
            var predicate = fixpoint.Predicate;
            var totalSet = new HashSet<RowWrapper>(new RowWrapperComparer(comparer));
            var totalRows = new List<object[]>();
            var context = parentContext ?? new EvaluationContext();
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

            while (context.Deltas.TryGetValue(predicate, out var delta) && delta.Count > 0)
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
                context.Deltas[predicate] = nextDelta;
            }

            return totalRows;
        }

        private IEnumerable<object[]> ExecuteMutualFixpoint(MutualFixpointNode node, EvaluationContext? parentContext)
        {
            if (node is null) throw new ArgumentNullException(nameof(node));

            var comparer = StructuralArrayComparer.Instance;
            var totalSets = new Dictionary<PredicateId, HashSet<RowWrapper>>();
            var context = parentContext ?? new EvaluationContext();

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

            while (context.Deltas.Values.Any(delta => delta.Count > 0))
            {
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
                                totalList.Add(tuple);
                                memberNext.Add(tuple);
                            }
                        }
                    }
                }

                foreach (var pair in nextDeltas)
                {
                    context.Deltas[pair.Key] = pair.Value;
                }
            }

            return context.Totals.TryGetValue(node.Head, out var headRows)
                ? headRows
                : Array.Empty<object[]>();
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
            public EvaluationContext(IEnumerable<object[]>? parameters = null, EvaluationContext? parent = null, QueryExecutionTrace? trace = null)
            {
                Parameters = parameters?.ToList() ?? new List<object[]>();
                Trace = trace ?? parent?.Trace;
                Facts = parent?.Facts ?? new Dictionary<PredicateId, List<object[]>>();
                FactSets = parent?.FactSets ?? new Dictionary<PredicateId, HashSet<object[]>>();
                FactIndices = parent?.FactIndices ?? new Dictionary<(PredicateId Predicate, int ColumnIndex), Dictionary<object, List<object[]>>>();
                JoinIndices = parent?.JoinIndices ?? new Dictionary<(PredicateId Predicate, string KeySignature), Dictionary<RowWrapper, List<object[]>>>();
            }

            public PredicateId Current { get; set; }
            = new PredicateId("", 0);

            public Dictionary<PredicateId, List<object[]>> Totals { get; } = new();

            public Dictionary<PredicateId, List<object[]>> Deltas { get; } = new();

            public IReadOnlyList<object[]> Parameters { get; }

            public QueryExecutionTrace? Trace { get; }

            public Dictionary<string, List<object[]>> Materialized { get; } = new();

            public Dictionary<PredicateId, List<object[]>> Facts { get; }

            public Dictionary<PredicateId, HashSet<object[]>> FactSets { get; }

            public Dictionary<(PredicateId Predicate, int ColumnIndex), Dictionary<object, List<object[]>>> FactIndices { get; }

            public Dictionary<(PredicateId Predicate, string KeySignature), Dictionary<RowWrapper, List<object[]>>> JoinIndices { get; }
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
