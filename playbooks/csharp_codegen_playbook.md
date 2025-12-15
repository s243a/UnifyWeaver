# Playbook: C# Codegen for Non-Recursive Predicates

## Audience
This playbook guides coding agents in using UnifyWeaver to compile non-recursive Prolog predicates to C# source code and execute them.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "csharp_codegen" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
python3 scripts/skills/lookup_example.py "how to use csharp codegen"


## Workflow Overview
This playbook demonstrates the compilation of a non-recursive predicate (`grandparent/2`) to C# and its execution.
1. Define Prolog facts (`parent/2`) and a rule (`grandparent/2`).
2. Compile the `grandparent/2` predicate to C# using the `csharp_codegen` target.
3. Execute the compiled C# program.
4. Verify the output.

## Agent Inputs
1. **Bash Executable Record** – `unifyweaver.execution.csharp_grandparent_bash` in `playbooks/examples_library/csharp_nonrecursive_examples.md`.
2. **PowerShell Executable Record** – `unifyweaver.execution.csharp_grandparent_ps` in `playbooks/examples_library/csharp_nonrecursive_examples.md`.
3. **Extraction Skill** – `skills/skill_extract_records.md`.

## Execution Guidance

An agent should choose the appropriate script based on the execution environment.

#### For Linux/macOS (bash) environment:

**Step 1: Navigate to project root**
```bash
cd /path/to/UnifyWeaver
```

**Step 2: Extract the bash script**
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.csharp_grandparent_bash" \
  --file-filter=all \
  playbooks/examples_library/csharp_nonrecursive_examples.md \
  > tmp/run_csharp_grandparent.sh
```

**Step 3: Make it executable**
```bash
chmod +x tmp/run_csharp_grandparent.sh
```

**Step 4: Run the bash script**
```bash
bash tmp/run_csharp_grandparent.sh
```

#### For Windows (PowerShell) environment:

**Step 1: Navigate to project root**
```powershell
cd C:\path\to\UnifyWeaver
```

**Step 2: Extract the PowerShell script**
```powershell
perl scripts/utils/extract_records.pl -f content -q "unifyweaver.execution.csharp_grandparent_ps" --file-filter=all playbooks/examples_library/csharp_nonrecursive_examples.md | Out-File -FilePath tmp/run_csharp_grandparent.ps1
```

**Step 3: Run the PowerShell script**
```powershell
./tmp/run_csharp_grandparent.ps1
```

**Expected Output**:
```
Compiling Prolog to C#...
Executing C# program...
anne:charles
anne:diana
Success: C# program compiled and executed successfully.
```

## Expected Outcome
- Successful execution will print "Success: C# program compiled and executed successfully."
- The output will contain the grandparent relationships: `anne:charles` and `anne:diana`
- Exit code 0.

## Platform-Specific Notes

### Linux/WSL/macOS (Bash)
The bash script includes:
- **Auto-detection of .NET SDK version** (checks for 8.0, 7.0, 6.0)
- **Proper here-document usage** for `.csproj` file creation
- **Cross-platform execution** using `dotnet run` instead of platform-specific `.exe`
- **Error handling** for missing .NET SDK

### Windows (PowerShell)
The PowerShell script includes:
- **Auto-detection of SWI-Prolog location** (checks common installation paths)
- **Auto-detection of .NET SDK version**
- **Cross-platform execution** using `dotnet run --no-build`
- **Comprehensive error messages** for missing dependencies

## Improvements Over Original
1. ✅ **Fixed**: `.exe` execution (now uses `dotnet run` for cross-platform compatibility)
2. ✅ **Fixed**: Hardcoded .NET version (now auto-detects available SDK)
3. ✅ **Fixed**: Hardcoded swipl path (PowerShell now auto-detects)
4. ✅ **Fixed**: Here-document issue in bash (uses proper `cat` syntax)
5. ✅ **Added**: Better error handling and diagnostics

## Troubleshooting

### "No compatible .NET SDK found"
**Cause**: .NET SDK 6.0, 7.0, or 8.0 not installed

**Solution**:
```bash
# Check installed SDKs
dotnet --list-sdks

# Install .NET (Ubuntu/Debian)
sudo apt-get install dotnet-sdk-8.0

# Install .NET (macOS)
brew install dotnet

# Install .NET (Windows)
# Download from https://dotnet.microsoft.com/download
```

### "Could not find swipl.exe" (PowerShell)
**Cause**: SWI-Prolog not in PATH or standard locations

**Solution**:
- Ensure SWI-Prolog is installed
- Add swipl to PATH, or
- Modify script to include your swipl location

### "arguments not sufficiently instantiated"
**Cause**: Issue with CSharpCode variable in Prolog compilation

**Solution**:
- Verify UnifyWeaver modules are properly loaded
- Check that `csharp_stream_target` module exports `compile_predicate_to_csharp/3`
- Ensure Prolog facts file loads correctly

