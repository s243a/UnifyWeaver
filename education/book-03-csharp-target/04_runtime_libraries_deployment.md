<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 4: Runtime Libraries and Deployment

## Overview

This chapter covers how to package, deploy, and integrate UnifyWeaver-generated C# code into production .NET applications.

## Runtime Architecture

The Query Runtime depends on a shared runtime library:

```
Your Application
    ↓ uses
UnifyWeaver.Runtime.dll  ← Shared query engine
    ↓ executes
QueryPlan (IR)  ← Generated from Prolog
    ↓ operates on
Data (facts, computed results)
```

### Components

**1. Core Runtime Library**
- `QueryEngine.cs` - Fixpoint iteration engine
- `IR.cs` - IR data structures (RelationRef, Join, etc.)
- `SemiNaiveEvaluator.cs` - Optimization logic
- `TupleStore.cs` - In-memory tuple storage

**2. Generated Code**
- Query plans (IR instances)
- Base fact data
- Entry points (Main methods)

**3. Your Application**
- Calls generated query methods
- Processes results
- Integrates with business logic

### Mutual Fixpoint Runtime Artifacts (v0.1)

To support mutual recursion the runtime now exposes additional types:

- `MutualFixpointNode` – top-level coordinator for strongly connected predicate groups.
- `MutualMember` – pairs a predicate with its base and recursive plan list.
- `CrossRefNode` – runtime reference to another predicate within the same SCC (delta/total aware).
- `EvaluationContext` – tracks per-predicate totals/deltas and the predicate currently being evaluated.
- `HashSet<object[]>` stores – provide Bash-parity deduplication for every predicate in the group.

Your generated plan will instantiate these nodes automatically when the compiler detects mutual recursion (e.g., `is_even/1` and `is_odd/1`). No manual changes are required, but you should ship the upgraded runtime library alongside your plans.

## Building Runtime Libraries

### Option 1: NuGet Package (Recommended)

Create a reusable NuGet package for the runtime:

**1. Create project:**
```bash
dotnet new classlib -n UnifyWeaver.Runtime
cd UnifyWeaver.Runtime
```

**2. Add runtime code:**

**IR.cs:**
```csharp
namespace UnifyWeaver.Runtime
{
    // RelationRef: Reference to a relation
    public class RelationRef<T>
    {
        public string Name { get; set; }
        public int Arity { get; set; }
        public IEnumerable<T> Source { get; set; }

        public IEnumerable<T> Evaluate() => Source;
    }

    // Selection: Filter operation
    public class Selection<T>
    {
        public IOperation<T> Source { get; set; }
        public Func<T, bool> Predicate { get; set; }

        public IEnumerable<T> Evaluate()
            => Source.Evaluate().Where(Predicate);
    }

    // Join: N-ary join operation
    public class Join<TOuter, TInner, TKey, TResult>
    {
        public IOperation<TOuter> Outer { get; set; }
        public IOperation<TInner> Inner { get; set; }
        public Func<TOuter, TKey> OuterKeySelector { get; set; }
        public Func<TInner, TKey> InnerKeySelector { get; set; }
        public Func<TOuter, TInner, TResult> ResultSelector { get; set; }

        public IEnumerable<TResult> Evaluate()
        {
            return Outer.Evaluate().Join(
                Inner.Evaluate(),
                OuterKeySelector,
                InnerKeySelector,
                ResultSelector
            );
        }
    }

    // Union: Combine operations
    public class Union<T>
    {
        public IOperation<T>[] Sources { get; set; }

        public IEnumerable<T> Evaluate()
            => Sources.SelectMany(s => s.Evaluate()).Distinct();
    }
}
```

**QueryEngine.cs:**
```csharp
namespace UnifyWeaver.Runtime
{
    public class QueryEngine
    {
        private Dictionary<string, IEnumerable<object>> _relations = new();

        // Register base facts
        public void RegisterRelation<T>(string name, IEnumerable<T> facts)
        {
            _relations[name] = facts.Cast<object>();
        }

        // Execute query with fixpoint iteration
        public IEnumerable<T> Execute<T>(QueryPlan<T> plan)
        {
            var current = new HashSet<T>();
            var changed = true;

            while (changed)
            {
                var before = current.Count;

                // Evaluate all clauses
                foreach (var clause in plan.Clauses)
                {
                    var results = clause.Evaluate();
                    foreach (var result in results)
                    {
                        current.Add(result);
                    }
                }

                changed = current.Count > before;
            }

            return current;
        }

        // Semi-naive evaluation (optimized)
        public IEnumerable<T> ExecuteSemiNaive<T>(QueryPlan<T> plan)
        {
            var allTuples = new HashSet<T>();
            var delta = new HashSet<T>();

            // Iteration 0: Base facts
            foreach (var baseFact in plan.BaseFacts)
            {
                allTuples.Add(baseFact);
                delta.Add(baseFact);
            }

            // Iterations 1+: Apply rules to delta
            while (delta.Any())
            {
                var newTuples = new HashSet<T>();

                foreach (var clause in plan.Clauses)
                {
                    // Only join with delta (new tuples)
                    var results = clause.EvaluateWithDelta(delta);

                    foreach (var result in results)
                    {
                        if (!allTuples.Contains(result))
                        {
                            newTuples.Add(result);
                            allTuples.Add(result);
                        }
                    }
                }

                delta = newTuples;
            }

            return allTuples;
        }
    }
}
```

**3. Build NuGet package:**
```bash
dotnet pack -c Release
# Generates: bin/Release/UnifyWeaver.Runtime.1.0.0.nupkg
```

**4. Publish to NuGet (optional):**
```bash
dotnet nuget push bin/Release/UnifyWeaver.Runtime.1.0.0.nupkg \
    --api-key YOUR_API_KEY \
    --source https://api.nuget.org/v3/index.json
```

### Option 2: Direct Project Reference

For single applications, reference the runtime project directly:

```bash
dotnet add reference ../UnifyWeaver.Runtime/UnifyWeaver.Runtime.csproj
```

## Integrating Generated Code

### Example: Family Tree Application

**1. Generate C# from Prolog:**

```prolog
% family.pl
:- use_module(unifyweaver(targets/csharp_query_target)).

parent(alice, bob).
parent(bob, charlie).
parent(charlie, dave).

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

% Compile to C#
?- compile_predicate_to_csharp(ancestor/2, [
    target(csharp_query),
    namespace('MyApp.Queries')
], Code),
   open('AncestorQuery.cs', write, S),
   write(S, Code),
   close(S).
```

**2. Generated code (simplified):**

**AncestorQuery.cs:**
```csharp
using UnifyWeaver.Runtime;

namespace MyApp.Queries
{
    public static class AncestorQuery
    {
        // Base facts
        static readonly (string, string)[] ParentFacts = {
            ("alice", "bob"),
            ("bob", "charlie"),
            ("charlie", "dave")
        };

        // Query plan
        public static IEnumerable<(string, string)> Execute()
        {
            var engine = new QueryEngine();

            // Register base facts
            engine.RegisterRelation("parent", ParentFacts);

            // Build query plan
            var plan = new QueryPlan<(string, string)> {
                BaseFacts = ParentFacts,
                Clauses = new[] {
                    // Clause 1: ancestor(X, Y) :- parent(X, Y)
                    new Clause<(string, string)> {
                        Body = new RelationRef<(string, string)> {
                            Name = "parent",
                            Source = ParentFacts
                        }
                    },

                    // Clause 2: ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
                    new Clause<(string, string)> {
                        Body = new Join<(string, string), (string, string), string, (string, string)> {
                            Outer = new RelationRef<(string, string)> {
                                Name = "parent",
                                Source = ParentFacts
                            },
                            Inner = new DynamicRelationRef<(string, string)> {
                                Name = "ancestor"  // Recursive reference
                            },
                            OuterKeySelector = t => t.Item2,
                            InnerKeySelector = t => t.Item1,
                            ResultSelector = (outer, inner) => (outer.Item1, inner.Item2)
                        }
                    }
                }
            };

            return engine.ExecuteSemiNaive(plan);
        }
    }
}
```

**3. Use in your application:**

**Program.cs:**
```csharp
using MyApp.Queries;

class Program
{
    static void Main()
    {
        var ancestors = AncestorQuery.Execute();

        foreach (var (ancestor, descendant) in ancestors)
        {
            Console.WriteLine($"{ancestor} is ancestor of {descendant}");
        }
    }
}
```

**Output:**
```
alice is ancestor of bob
alice is ancestor of charlie
alice is ancestor of dave
bob is ancestor of charlie
bob is ancestor of dave
charlie is ancestor of dave
```

## Deployment Scenarios

### Scenario 1: Console Application

**Structure:**
```
MyConsoleApp/
├── MyConsoleApp.csproj
├── Program.cs
├── Queries/
│   ├── AncestorQuery.cs  (generated)
│   └── SiblingQuery.cs   (generated)
└── UnifyWeaver.Runtime/  (NuGet package or project ref)
```

**Publish:**
```bash
dotnet publish -c Release -r linux-x64 --self-contained
# Output: bin/Release/net6.0/linux-x64/publish/
```

**Deploy:**
```bash
scp -r bin/Release/net6.0/linux-x64/publish/ server:/opt/myapp/
ssh server '/opt/myapp/MyConsoleApp'
```

### Scenario 2: ASP.NET Web API

**Structure:**
```
MyWebApi/
├── MyWebApi.csproj
├── Controllers/
│   └── FamilyController.cs
├── Queries/
│   └── AncestorQuery.cs  (generated)
└── UnifyWeaver.Runtime/
```

**Controller:**
```csharp
using Microsoft.AspNetCore.Mvc;
using MyApp.Queries;

[ApiController]
[Route("api/family")]
public class FamilyController : ControllerBase
{
    [HttpGet("ancestors/{person}")]
    public IActionResult GetAncestors(string person)
    {
        var ancestors = AncestorQuery.Execute()
            .Where(t => t.Item2 == person)
            .Select(t => t.Item1);

        return Ok(ancestors);
    }
}
```

**Usage:**
```bash
curl http://localhost:5000/api/family/ancestors/dave
# Returns: ["alice", "bob", "charlie"]
```

### Scenario 3: Azure Functions

**Function:**
```csharp
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Http;
using MyApp.Queries;

public class FamilyFunctions
{
    [Function("GetAncestors")]
    public HttpResponseData GetAncestors(
        [HttpTrigger(AuthorizationLevel.Anonymous, "get")] HttpRequestData req,
        string person)
    {
        var ancestors = AncestorQuery.Execute()
            .Where(t => t.Item2 == person)
            .Select(t => t.Item1);

        var response = req.CreateResponse(HttpStatusCode.OK);
        response.WriteAsJsonAsync(ancestors);
        return response;
    }
}
```

### Scenario 4: Desktop Application (WPF/WinForms)

**ViewModel:**
```csharp
using System.Collections.ObjectModel;
using MyApp.Queries;

public class FamilyTreeViewModel
{
    public ObservableCollection<Person> Ancestors { get; set; }

    public void LoadAncestors(string person)
    {
        var ancestors = AncestorQuery.Execute()
            .Where(t => t.Item2 == person)
            .Select(t => new Person { Name = t.Item1 });

        Ancestors = new ObservableCollection<Person>(ancestors);
    }
}
```

## Cross-Platform Deployment

### Target Runtime Identifiers (RIDs)

**Windows:**
```bash
dotnet publish -r win-x64 --self-contained
dotnet publish -r win-arm64  # ARM Windows
```

**Linux:**
```bash
dotnet publish -r linux-x64 --self-contained
dotnet publish -r linux-arm64  # Raspberry Pi, etc.
```

**macOS:**
```bash
dotnet publish -r osx-x64 --self-contained
dotnet publish -r osx-arm64  # Apple Silicon
```

### Framework-Dependent vs Self-Contained

**Framework-Dependent (smaller, requires .NET runtime):**
```bash
dotnet publish -c Release
# Output: ~200KB (app only)
# Requires: .NET 6.0+ runtime on target
```

**Self-Contained (larger, no dependencies):**
```bash
dotnet publish -c Release -r linux-x64 --self-contained
# Output: ~60MB (includes .NET runtime)
# Requires: Nothing! Fully standalone
```

**Single-File Executable:**
```bash
dotnet publish -c Release -r linux-x64 \
    --self-contained \
    -p:PublishSingleFile=true
# Output: Single executable file
```

## Performance Tuning

### 1. Precompute Static Queries

For queries that don't change:

```csharp
public static class AncestorQuery
{
    // Compute once at startup
    private static readonly Lazy<List<(string, string)>> _cachedResults
        = new Lazy<List<(string, string)>>(() => Execute().ToList());

    public static IEnumerable<(string, string)> GetCached()
        => _cachedResults.Value;
}
```

### 2. Parallel Evaluation

For independent clauses:

```csharp
public IEnumerable<T> ExecuteParallel<T>(QueryPlan<T> plan)
{
    var results = plan.Clauses
        .AsParallel()
        .SelectMany(clause => clause.Evaluate())
        .Distinct();

    return results;
}
```

### 3. Database Backend

For large fact sets, use a database:

```csharp
public class DatabaseBackedRelation<T> : IRelation<T>
{
    private readonly string _connectionString;
    private readonly string _tableName;

    public IEnumerable<T> Evaluate()
    {
        using var connection = new SqlConnection(_connectionString);
        var query = $"SELECT * FROM {_tableName}";
        return connection.Query<T>(query);  // Using Dapper
    }
}
```

### 4. Streaming Results

For very large result sets:

```csharp
public IAsyncEnumerable<T> ExecuteAsync<T>(QueryPlan<T> plan)
{
    await foreach (var result in plan.EvaluateStreamingAsync())
    {
        yield return result;
    }
}
```

## Monitoring and Logging

### Add Instrumentation

```csharp
public class InstrumentedQueryEngine : QueryEngine
{
    private readonly ILogger _logger;

    public override IEnumerable<T> Execute<T>(QueryPlan<T> plan)
    {
        var stopwatch = Stopwatch.StartNew();

        _logger.LogInformation("Starting query: {PlanName}", plan.Name);

        var results = base.Execute(plan).ToList();

        stopwatch.Stop();

        _logger.LogInformation(
            "Query {PlanName} completed in {ElapsedMs}ms, {ResultCount} results",
            plan.Name,
            stopwatch.ElapsedMilliseconds,
            results.Count
        );

        return results;
    }
}
```

### Metrics

```csharp
public class MetricsQueryEngine : QueryEngine
{
    public QueryMetrics GetMetrics(QueryPlan plan)
    {
        return new QueryMetrics {
            TotalIterations = _iterationCount,
            TotalTuplesGenerated = _tupleCount,
            FixpointReachedAt = _fixpointIteration,
            ExecutionTime = _executionTime
        };
    }
}
```

## Testing Generated Code

### Unit Tests

```csharp
using Xunit;
using MyApp.Queries;

public class AncestorQueryTests
{
    [Fact]
    public void Execute_ShouldReturnAllAncestors()
    {
        var ancestors = AncestorQuery.Execute().ToList();

        Assert.Contains(("alice", "bob"), ancestors);
        Assert.Contains(("alice", "charlie"), ancestors);
        Assert.Contains(("alice", "dave"), ancestors);
        Assert.Contains(("bob", "charlie"), ancestors);
        Assert.Contains(("bob", "dave"), ancestors);
        Assert.Contains(("charlie", "dave"), ancestors);
    }

    [Fact]
    public void Execute_ShouldNotContainSelfAncestry()
    {
        var ancestors = AncestorQuery.Execute();

        Assert.DoesNotContain(
            ancestors,
            t => t.Item1 == t.Item2
        );
    }
}
```

### Integration Tests

```csharp
public class FamilyApiTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public FamilyApiTests(WebApplicationFactory<Program> factory)
    {
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task GetAncestors_ShouldReturnCorrectResults()
    {
        var response = await _client.GetAsync("/api/family/ancestors/dave");
        response.EnsureSuccessStatusCode();

        var ancestors = await response.Content.ReadAsAsync<List<string>>();

        Assert.Contains("alice", ancestors);
        Assert.Contains("bob", ancestors);
        Assert.Contains("charlie", ancestors);
    }
}
```

## Continuous Integration

### GitHub Actions Example

**.github/workflows/build.yml:**
```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Setup .NET
      uses: actions/setup-dotnet@v1
      with:
        dotnet-version: 6.0.x

    - name: Setup SWI-Prolog
      run: |
        sudo apt-get update
        sudo apt-get install -y swi-prolog

    - name: Generate C# from Prolog
      run: |
        cd src/prolog
        swipl -g "compile_all, halt" -t 'halt(1)'

    - name: Restore dependencies
      run: dotnet restore

    - name: Build
      run: dotnet build --no-restore

    - name: Test
      run: dotnet test --no-build --verbosity normal

    - name: Publish
      run: dotnet publish -c Release -o publish/

    - name: Upload artifacts
      uses: actions/upload-artifact@v2
      with:
        name: app
        path: publish/
```

## Summary

This chapter covered:
- ✅ Building runtime libraries (NuGet packages)
- ✅ Integrating generated code into applications
- ✅ Deployment scenarios (console, web, desktop, cloud)
- ✅ Cross-platform publishing
- ✅ Performance tuning strategies
- ✅ Testing and CI/CD

**You now have everything you need to:**
- Compile Prolog to C#
- Package and deploy .NET applications
- Integrate with existing systems
- Monitor and optimize performance

## Testing & Validation (v0.1)

- Run the automated regression with the skip flag:
  ```bash
  SKIP_CSHARP_EXECUTION=1 swipl -q \
      -f tests/core/test_csharp_query_target.pl \
      -g test_csharp_query_target:test_csharp_query_target \
      -t halt
  ```
- Optionally execute the full suite `run_all_tests.pl` the same way.
- For end-to-end verification, follow the build-first flow described in the [C# Query Target Test Plan](https://github.com/s243a/UnifyWeaver/blob/main/docs/development/testing/v0_1_csharp_test_plan.md) to emit a project under `output/csharp/<uuid>/`, run `dotnet build --no-restore`, and execute the compiled binary/DLL.

Document your results (e.g., include command transcripts in release notes) so future releases can reproduce the validation steps.

---

**Final Exercises:**

1. Create a NuGet package for the runtime library
2. Build a simple web API using generated queries
3. Deploy to Azure or AWS
4. Set up CI/CD pipeline for your project

## What's Next?

**Extend your knowledge:**
- Explore custom IR operations
- Implement database-backed relations
- Add caching layers
- Profile and optimize complex queries

**Contribute:**
- Improve runtime library
- Add new target languages
- Share your use cases

**Get Help:**
- GitHub Discussions: https://github.com/s243a/UnifyWeaver/discussions
- Issues: https://github.com/s243a/UnifyWeaver/issues

---

**Congratulations!** You've completed Book 2: C# Target Language.

Return to Book 1 for Bash-specific topics or explore the appendices for advanced patterns.

---

## Navigation

**←** [Previous: Chapter 3: C# Query Runtime - Deep Dive](03_query_engine_deep_dive) | [📖 Book 3: C# Target](README) | [Next: Chapter 5: Semantic Crawling and Vector Search →](05_semantic_crawling)
