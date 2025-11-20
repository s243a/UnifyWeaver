// SPDX-License-Identifier: MIT OR Apache-2.0
// Generated runtime scaffolding for UnifyWeaver C# query execution
// Provides minimal infrastructure for executing declarative plans
// emitted by the forthcoming csharp_query target.

using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Globalization;

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
                return JsonSerializer.Deserialize(json, _targetType, _serializerOptions)
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
                            foreach (var property in element.EnumerateObject())
                            {
                                output.Add(property.Value);
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
