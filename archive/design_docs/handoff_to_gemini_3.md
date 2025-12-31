**Handoff Document for Gemini 3.0**

**Problem:**
The primary goal is to successfully execute the `json_litedb_playbook.md` playbook. This playbook's PowerShell example (`unifyweaver.execution.json_to_litedb_ps`) fails during the C# compilation/loading step. The underlying issue appears to be a series of environment-related problems with PowerShell's `Add-Type` cmdlet, and with the UnifyWeaver `dotnet_source.pl` module's ability to handle C# compilation robustly.

**Last Known State:**
The execution of the PowerShell script generated from the playbook (`tmp/run_json_litedb_example.ps1`) consistently fails. The errors encountered include:
1.  **C# Compilation Error:** `Failed to compile C# code: (0) : Metadata file 'System.Text.Json.dll' could not be found`. This error occurs even when the full path to the DLL is provided.
2.  **C# Language Version Error:** `Unexpected character '$'` when using C# 6.0+ interpolated strings.
3.  **C# Syntax Error:** `Invalid expression term '['` when using C# 6.0+ collection initializer syntax.
4.  **Assembly Collision:** `The predefined type 'System.Byte' is defined in multiple assemblies...` when explicitly referencing .NET Core assemblies like `System.Private.CoreLib.dll`.
5.  **Prolog Syntax Errors:** Various Prolog syntax errors have occurred while attempting to modify `dotnet_source.pl` to work around the above issues.

**What has been tried:**
*   **Explicitly referencing .NET Core assemblies:** I have tried explicitly referencing `System.Text.Json.dll`, `System.Runtime.dll`, `System.Private.CoreLib.dll`, `Microsoft.CSharp.dll`, `System.Linq.dll`, and `System.Linq.Expressions.dll` with their full paths. This leads to the "type is defined in multiple assemblies" error.
*   **Using `string.Format()`:** I have modified the C# code to use `string.Format()` instead of interpolated strings, and explicit `BsonDocument` initialization. This resolved the syntax errors but not the assembly collision or "metadata file not found" errors.
*   **Modifying `dotnet_source.pl`:** I have attempted to modify `src/unifyweaver/sources/dotnet_source.pl` to:
    *   Use `/nostdlib` with `Add-Type -CompilerOptions`, but the `-CompilerOptions` parameter is not supported in the current environment.
    *   Introduce an `external_compile(true)` option and a new template to use `dotnet build` externally. This has proven difficult due to the complexity of the template system and the need for correct Prolog syntax and escaping.

**Current State of `src/unifyweaver/sources/dotnet_source.pl`:**
I have reverted `src/unifyweaver/sources/dotnet_source.pl` to its original state.

**Current State of `playbooks/examples_library/json_litedb_examples.md`:**
The `unifyweaver.execution.json_to_litedb_ps` record in this file has been modified to use modern C# syntax.

**Suggested next steps for Gemini 3.0:**
1.  **Do not run the LiteDB setup script.** Assume the libraries are correctly installed as per the user's instructions.
2.  **Focus on the `dotnet_source.pl` refactoring.** The `Add-Type -TypeDefinition` approach has been exhausted and is a dead end. The `dotnet build` approach is the correct path.
3.  **Verify the `dotnet_source_external_compile` template logic.** The template should:
    *   Generate a `.csproj` file that includes a `PackageReference` for `LiteDB`.
    *   Generate a `.cs` file with the C# code.
    *   Generate PowerShell code that runs `dotnet build`.
    *   **Crucially, before calling `Add-Type -Path` on the newly compiled DLL, the PowerShell code must *first* load its dependencies.** Specifically, it must call `Add-Type -Path` on `lib/LiteDB.dll`. The "Failed to load compiled DLL" error strongly suggests this is the missing piece.
4.  **Carefully modify `src/unifyweaver/sources/dotnet_source.pl`.** Given my repeated failures, I recommend you:
    *   Read the file carefully.
    *   Use a very specific, multi-line `old_string` for any `replace` operations to ensure you are targeting the correct part of the template.
    *   Add a pre-loading loop for `{{dll_references}}` in the `dotnet_source_external_compile` template before the final `Add-Type -Path $compiledDll` call.
5.  **Update the example.** Ensure the `unifyweaver.execution.json_to_litedb_ps` example in `playbooks/examples_library/json_litedb_examples.md` uses `external_compile(true)` and modern C# syntax.

This problem is solvable with the external compilation approach. My difficulties were in the execution of the file modifications, not the overall strategy.