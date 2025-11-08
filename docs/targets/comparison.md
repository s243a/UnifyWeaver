# Target Comparison Matrix

The table below contrasts the Bash backend with the two C# variants discussed in the accompanying documents. Use it to choose the appropriate target for each deployment scenario.

| Capability / Feature                  | Bash (`target(bash)`)                    | C# Codegen (`target(csharp_codegen)`)                    | C# Query Runtime (`target(csharp_query)`)                          |
|--------------------------------------|------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------------|
| Artefact form                        | Shell script                             | C# source files                                         | Query IR + shared engine                                          |
| Runtime dependency                   | POSIX shell, coreutils                   | .NET runtime + compiler                                 | .NET runtime (engine library)                                     |
| Non-recursive support                | ✅ Mature                                 | ✅ Implemented                                          | ✅ Implemented                                                     |
| Recursive support                    | ✅ Memoised + advanced patterns          | ⚠️ Not yet (falls back)                                 | ✅ Semi-naive fixpoint (linear + mutual), advanced patterns TBD   |
| Dedup strategies                     | `sort`, `declare -A`, manual loop        | `Distinct()`, `HashSet<T>` (partial parity)             | `HashSet<object[]>` (ordered dedup planned)                       |
| Performance profile                  | Good for small/medium, I/O bound         | Strong for in-memory workloads                          | Tunable; suited for iterative evaluation                          |
| Debuggability                        | Inspect shell scripts                    | Step through C# in IDE                                  | Instrumented engine logs iterations                               |
| Integration                          | Shell pipelines, CLI tooling             | Embed in .NET apps, publish NuGet packages              | Reusable engine, potential for distributed execution              |
| Security posture                     | Commands visible; firewall can block ops | Managed code; sandbox through firewall + CAS policies   | Managed code; IR inspection enables policy checks                 |
| Best use cases                       | Quick prototyping, Unix-native workflows | Managed services, applications needing static artefacts | Complex recursion, dynamic optimisation, gradual runtime upgrades |

## Combined `target(csharp)` Behaviour
To keep user experience simple, the generic `target(csharp)` option will behave as a smart facade:
1. Attempt `csharp_codegen` for predicates and features the codegen backend supports.
2. Fall back to `csharp_query` when recursion or other unsupported patterns appear.
3. Provide diagnostics explaining which backend handled each predicate so operators can tune preferences.

This approach gives immediate value from the existing code generator while allowing incremental rollout of the query runtime without breaking existing scripts.
