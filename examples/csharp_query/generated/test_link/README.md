# C# Query Target Output (test_link)

This directory captures the exact C# artifacts currently emitted by the `csharp_query` target when compiling the simple two-clause `test_link/2` predicate used in our unit tests.  The files are copied verbatim from the harness output after running `test_csharp_query_target/0`, with only the `bin/` and `obj/` build directories removed.

Contents:
- `test_link_example.csproj` – the console project created via `dotnet new console --framework net9.0`
- `Program.cs` – the minimalist harness that executes the generated plan through `QueryExecutor`
- `QueryRuntime.cs` – the shared runtime library bundled alongside each generated project
- `TestLinkQueryModule.cs` – the plan/module produced by the compiler for `test_link/2`

These artifacts are read-only reference material; they are not wired into the build.  Regenerate them by running the C# query target tests with `--csharp-query-keep` and copying a fresh output directory into `examples/csharp_query/generated/`.
