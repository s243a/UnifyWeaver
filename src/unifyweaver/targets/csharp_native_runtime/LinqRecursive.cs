// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2025 UnifyWeaver Contributors
//
// LinqRecursive - LINQ-style extension methods for recursive queries
// Uses semi-naive iteration for efficient fixpoint computation

using System;
using System.Collections.Generic;
using System.Linq;

namespace UnifyWeaver.Native
{
    /// <summary>
    /// Provides LINQ-style extension methods for recursive query patterns.
    /// These methods use semi-naive iteration internally for efficient
    /// fixpoint computation while preserving familiar LINQ semantics.
    /// </summary>
    public static class LinqRecursive
    {
        /// <summary>
        /// Computes the transitive closure of a relation using semi-naive iteration.
        ///
        /// Example - ancestor/2:
        ///   ancestor(X,Y) :- parent(X,Y).
        ///   ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z).
        ///
        /// Usage:
        ///   ParentStream().TransitiveClosure(
        ///       parents => parents,
        ///       (ancestor, parents) => parents
        ///           .Where(p => p.Item2 == ancestor.Item1)
        ///           .Select(p => (p.Item1, ancestor.Item2))
        ///   )
        /// </summary>
        /// <typeparam name="T">The tuple type of the relation</typeparam>
        /// <param name="baseRelation">The base relation (e.g., parent facts)</param>
        /// <param name="seedSelector">Function to derive initial results from base relation</param>
        /// <param name="expandStep">Function that joins delta tuples with base relation to produce new tuples</param>
        /// <returns>All tuples in the transitive closure</returns>
        public static IEnumerable<T> TransitiveClosure<T>(
            this IEnumerable<T> baseRelation,
            Func<IEnumerable<T>, IEnumerable<T>> seedSelector,
            Func<T, IEnumerable<T>, IEnumerable<T>> expandStep)
        {
            // Materialize base relation to avoid multiple enumeration
            var baseList = baseRelation.ToList();
            var seen = new HashSet<T>();
            var delta = new List<T>();

            // Seed from base case
            foreach (var item in seedSelector(baseList))
            {
                if (seen.Add(item))
                {
                    delta.Add(item);
                    yield return item;
                }
            }

            // Semi-naive iteration: only process new delta tuples
            while (delta.Count > 0)
            {
                var newDelta = new List<T>();
                foreach (var d in delta)
                {
                    foreach (var result in expandStep(d, baseList))
                    {
                        if (seen.Add(result))
                        {
                            newDelta.Add(result);
                            yield return result;
                        }
                    }
                }
                delta = newDelta;
            }
        }

        /// <summary>
        /// Simplified transitive closure for common pattern where seed is the base relation itself.
        ///
        /// Usage:
        ///   ParentStream().TransitiveClosure(
        ///       (ancestor, parents) => parents
        ///           .Where(p => p.Item2 == ancestor.Item1)
        ///           .Select(p => (p.Item1, ancestor.Item2))
        ///   )
        /// </summary>
        public static IEnumerable<T> TransitiveClosure<T>(
            this IEnumerable<T> baseRelation,
            Func<T, IEnumerable<T>, IEnumerable<T>> expandStep)
        {
            return baseRelation.TransitiveClosure(seed => seed, expandStep);
        }

        /// <summary>
        /// Safe recursive join that prevents stack overflow by using iteration.
        /// Similar to LINQ Join but safe for self-referential queries.
        ///
        /// Example - reachable/2:
        ///   edge(a, b). edge(b, c). edge(c, d).
        ///   reachable(X, Y) :- edge(X, Y).
        ///   reachable(X, Z) :- edge(X, Y), reachable(Y, Z).
        ///
        /// Usage:
        ///   EdgeStream().SafeRecursiveJoin(
        ///       edges => edges,                           // seed
        ///       (reachable, edges) => edges               // expand
        ///           .Where(e => e.Item2 == reachable.Item1)
        ///           .Select(e => (e.Item1, reachable.Item2)),
        ///       result => result                          // project
        ///   )
        /// </summary>
        public static IEnumerable<TResult> SafeRecursiveJoin<TBase, TResult>(
            this IEnumerable<TBase> baseRelation,
            Func<IEnumerable<TBase>, IEnumerable<TResult>> seedSelector,
            Func<TResult, IEnumerable<TBase>, IEnumerable<TResult>> expandStep,
            Func<TResult, TResult> resultSelector)
        {
            var baseList = baseRelation.ToList();
            var seen = new HashSet<TResult>();
            var delta = new List<TResult>();

            // Seed
            foreach (var item in seedSelector(baseList))
            {
                var result = resultSelector(item);
                if (seen.Add(result))
                {
                    delta.Add(result);
                    yield return result;
                }
            }

            // Iterate
            while (delta.Count > 0)
            {
                var newDelta = new List<TResult>();
                foreach (var d in delta)
                {
                    foreach (var item in expandStep(d, baseList))
                    {
                        var result = resultSelector(item);
                        if (seen.Add(result))
                        {
                            newDelta.Add(result);
                            yield return result;
                        }
                    }
                }
                delta = newDelta;
            }
        }

        /// <summary>
        /// Computes fixpoint of an expansion function starting from a seed.
        /// Most general form - use when TransitiveClosure pattern doesn't fit.
        ///
        /// Usage:
        ///   LinqRecursive.Fixpoint(
        ///       initialFacts,
        ///       current => DeriveNewFacts(current, baseData)
        ///   )
        /// </summary>
        public static IEnumerable<T> Fixpoint<T>(
            IEnumerable<T> seed,
            Func<IEnumerable<T>, IEnumerable<T>> expand)
        {
            var seen = new HashSet<T>();
            var delta = new List<T>();

            foreach (var item in seed)
            {
                if (seen.Add(item))
                {
                    delta.Add(item);
                    yield return item;
                }
            }

            while (delta.Count > 0)
            {
                var newDelta = new List<T>();
                foreach (var item in expand(delta))
                {
                    if (seen.Add(item))
                    {
                        newDelta.Add(item);
                        yield return item;
                    }
                }
                delta = newDelta;
            }
        }

        /// <summary>
        /// Memoized version of TransitiveClosure that caches results for repeated calls.
        /// Thread-safe for concurrent access.
        /// </summary>
        public static Func<IEnumerable<T>> MemoizedTransitiveClosure<T>(
            Func<IEnumerable<T>> baseRelationFactory,
            Func<IEnumerable<T>, IEnumerable<T>> seedSelector,
            Func<T, IEnumerable<T>, IEnumerable<T>> expandStep)
        {
            List<T>? cached = null;
            object lockObj = new object();

            return () =>
            {
                if (cached != null) return cached;

                lock (lockObj)
                {
                    if (cached != null) return cached;
                    cached = baseRelationFactory()
                        .TransitiveClosure(seedSelector, expandStep)
                        .ToList();
                    return cached;
                }
            };
        }
    }
}
