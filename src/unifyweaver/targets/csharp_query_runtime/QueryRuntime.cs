// SPDX-License-Identifier: MIT OR Apache-2.0
// Generated runtime scaffolding for UnifyWeaver C# query execution
// Provides minimal infrastructure for executing declarative plans
// emitted by the forthcoming csharp_query target.

using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;

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

                case ArithmeticNode arithmetic:
                    return ExecuteArithmetic(arithmetic, context);

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

                case MutualFixpointNode mutualFixpoint:
                    return ExecuteMutualFixpoint(mutualFixpoint);

                case RecursiveRefNode recursiveRef:
                    return EvaluateRecursiveReference(recursiveRef, context);

                case CrossRefNode crossRef:
                    return EvaluateCrossReference(crossRef, context);

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

        private IEnumerable<object[]> ExecuteFixpoint(FixpointNode fixpoint)
        {
            if (fixpoint is null) throw new ArgumentNullException(nameof(fixpoint));

            var comparer = StructuralArrayComparer.Instance;
            var predicate = fixpoint.Predicate;
            var totalSet = new HashSet<RowWrapper>(new RowWrapperComparer(comparer));
            var totalRows = new List<object[]>();
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

            var context = new EvaluationContext
            {
                Current = predicate
            };
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

        private IEnumerable<object[]> ExecuteMutualFixpoint(MutualFixpointNode node)
        {
            if (node is null) throw new ArgumentNullException(nameof(node));

            var comparer = StructuralArrayComparer.Instance;
            var totalSets = new Dictionary<PredicateId, HashSet<RowWrapper>>();
            var context = new EvaluationContext();

            foreach (var member in node.Members)
            {
                var predicate = member.Predicate;
                var totalList = new List<object[]>();
                var deltaList = new List<object[]>();
                context.Totals[predicate] = totalList;
                context.Deltas[predicate] = deltaList;

                var set = new HashSet<RowWrapper>(new RowWrapperComparer(comparer));
                totalSets[predicate] = set;

                var baseRows = Evaluate(member.BasePlan).ToList();
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
            public PredicateId Current { get; set; }
            = new PredicateId("", 0);

            public Dictionary<PredicateId, List<object[]>> Totals { get; } = new();

            public Dictionary<PredicateId, List<object[]>> Deltas { get; } = new();
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
            using var reader = new StreamReader(_config.InputPath, Encoding.UTF8);
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
            using var stream = File.OpenRead(_config.InputPath);
            using var reader = new StreamReader(stream, Encoding.UTF8);
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
    }
}
