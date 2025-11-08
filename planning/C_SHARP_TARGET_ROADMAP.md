# C# Target Roadmap

## Objectives

- Grow the LINQ-based streaming compiler into a first-class target that can coexist with Bash and PowerShell backends.
- Define file/project layout conventions for generated C# assets (sources, data, build outputs).
- Plan communication and deployment models that allow C# components to interoperate with other runtime targets.
- Introduce a **location** abstraction so pipelines can span filesystems, machines, or network boundaries without leaking implementation details.

## Design Pillars

### 1. Source & Assembly Layout

- **Single-file scripts (current)** – keep the existing self-contained file for quick experiments and CLI usage.
- **Project-oriented generation** – emit `.csproj` scaffolding, namespaces, and folder structure when the user opts into a multi-file output mode.
  - Support splitting helpers (`DataSources.cs`, `Pipelines.cs`, `Program.cs`) for larger rule sets.
  - Allow embedding or referencing resource files (e.g. JSON data, CSV seeds).
- **Build tooling** – define how we invoke `dotnet build` or `dotnet script` during compilation/testing.
  - Provide knobs to produce executable binaries, DLLs, or NuGet packages depending on deployment needs.

### 2. Location & Deployment Model

- Introduce a `location(Type, Identifier, Properties)` abstraction.
  - `file(Path)`, `directory(Path)`, `machine(Host)`, `container(Name)`, `service(Endpoint)`.
- Annotate generated streams with their location requirements (e.g. `local_memory`, `service://inventory-api`).
- For distributed deployments, encode how data flows between locations (e.g. via REST call, message queue, shared storage).
- Ensure the planner can resolve location compatibility when mixing targets (Bash process might write to `tmp`; C# worker reads via file watch or HTTP).

### 3. Cross-Target Interop

- **Serialization contract** – prefer line-delimited JSON or a lightweight envelope (`{ "a": ..., "b": ... }`) to transport tuples between runtimes.
- **Process orchestration** – describe how a Bash or PowerShell stage invokes a compiled C# binary (CLI arguments, stdin/stdout, gRPC).
- **Proxy/adapter layer** – explore generating adapters that convert textual streams into typed objects (and vice versa) so each target can speak a common protocol.
- **Error handling** – define how exceptions or exit codes propagate across target boundaries with consistent diagnostics.

### 4. Expanded Streaming Semantics

- Support non-binary tuples by generating custom record types or using `ValueTuple` of higher arity.
- Expose asynchronous pipelines using `IAsyncEnumerable<T>` when upstream/downstream targets benefit from async boundaries.
- Allow stateful or windowed operations by mapping Prolog constructs to LINQ operators (`GroupBy`, `Aggregate`).
- Provide optional materialization points (write-to-disk, buffer, queue) when location changes require persistence.

## Implementation Phases

1. **Compiler completeness** – add fact-only support, multi-clause orchestration, and non-binary handling in the existing single-file flow.
2. **Project scaffolding** – introduce options for `.csproj`/folder output and integrate with template system.
3. **Interop adapters** – implement JSON line streaming and CLI invocation helpers so Bash and PowerShell targets can call into C# components.
4. **Location metadata** – extend the planner/configuration system to capture location requirements and validate cross-target pipelines.
5. **Distributed flows** – add runtime shims (e.g. HTTP proxy, message queue bridge) to move data across machines or services.

## Open Questions

- How should authentication/secrets be handled when a C# component calls remote services as part of a pipeline?
- What build/runtime dependencies are acceptable (pure .NET SDK vs. external libraries)?
- How do we represent versioning and compatibility across generated assemblies when the schema evolves?
- Should we standardize on `dotnet script` (single-file) or `dotnet build` (projects) as the default execution model?

## Next Actions

1. Finalize compiler enhancements (facts + multi-clause) to unblock richer documentation examples.
2. Prototype JSON line serialization between Bash→C# and C#→PowerShell for a simple pipeline.
3. Draft a schema for the `location/3` abstraction and integrate it into configuration preferences.
4. Evaluate template engine needs for `.csproj` and multi-file emission.
5. Document operational runbooks (build, deploy, monitor) once integration patterns stabilize.
