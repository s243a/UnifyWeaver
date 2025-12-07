---
id: playbook-gnu-prolog-compile-v1
name: "GNU Prolog Native Binary Compilation Playbook"
version: 1.0
workflow_id: prolog-target-compile
pattern: prolog_compilation
target: gnu_prolog_binary
difficulty: intermediate
estimated_time: 10_minutes
prerequisites: [swipl, unifyweaver_installed, gprolog, gplc]
related_books: [book-prolog-target]
---

# GNU Prolog Native Binary Compilation Playbook

## Goal

Use UnifyWeaver's Prolog target to generate a GNU Prolog script from a factorial predicate, compile it to a native binary using `gplc`, and verify the binary executes correctly without requiring a Prolog interpreter.

## Context

This playbook demonstrates **Prolog-to-Prolog transpilation** with native compilation - a powerful feature that enables:

- Generating standalone executables from Prolog predicates
- Fast startup times (no interpreter loading)
- Portable binaries for deployment
- Cross-dialect development (write in SWI, deploy as GNU)

**Why transpile Prolog to Prolog?**

While it may seem circular, transpiling Prolog to GNU Prolog provides:
1. **Dialect translation**: SWI-Prolog → GNU Prolog syntax
2. **Dependency injection**: Automatic inclusion of required modules
3. **Initialization handling**: Correct entry points for compiled vs interpreted modes
4. **Compilation support**: Automatic invocation of `gplc` compiler
5. **Error handling**: Graceful fallback if compilation fails

**Assumptions**:
- UnifyWeaver is installed at `$UNIFYWEAVER_HOME`
- SWI-Prolog (`swipl`) is available
- GNU Prolog (`gprolog`) is installed
- GNU Prolog compiler (`gplc`) is available in PATH
- You have write access to `/tmp/` and current directory

**Background**:
The Prolog target (covered in book-prolog-target) handles:
- Dialect-specific code generation
- Correct initialization directives for compiled binaries (`:- initialization(Goal).`)
- Compilation error checking with exit code validation
- Graceful fallback to interpreted mode if compilation fails

See **[Book: Prolog Target](../../book-prolog-target/README.md)** for implementation details.

## Strategies

### Strategy A: Direct Compilation (Recommended)
- **Cost**: Low (single step)
- **Speed**: Fast compilation (~2-5 seconds)
- **Quality**: Standard executable
- **When to use**: Standard deployment, small predicates

### Strategy B: Compilation with Fallback Testing
- **Cost**: Medium (tests both compiled and interpreted)
- **Speed**: Slower (~10-15 seconds total)
- **Quality**: High (validates both modes)
- **When to use**: Critical deployments, cross-platform code

**Heuristic**: Use Strategy A for development, Strategy B for production releases.

## Execution Steps

### Step 1: Define Factorial Predicate

Create `/tmp/factorial_gnu.pl` with a simple factorial predicate:

```bash
cat > /tmp/factorial_gnu.pl <<'EOF'
% factorial(+N, -Result) - Compute factorial of N
factorial(0, 1) :- !.
factorial(N, Result) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, PrevResult),
    Result is N * PrevResult.

% test_factorial/0 - Test with known values
test_factorial :-
    factorial(5, F5),
    format('Factorial of 5 is ~w~n', [F5]),
    factorial(10, F10),
    format('Factorial of 10 is ~w~n', [F10]).
EOF
```

**Verification**: File should exist with 14 lines
```bash
test -f /tmp/factorial_gnu.pl && wc -l /tmp/factorial_gnu.pl
# Expected: 14 /tmp/factorial_gnu.pl
```

### Step 2: Test in SWI-Prolog First

Always verify your Prolog code works before transpiling:

```bash
swipl -q -f /tmp/factorial_gnu.pl -g "test_factorial, halt"
```

**Expected output**:
```
Factorial of 5 is 120
Factorial of 10 is 3628800
```

> [!tip]
> **Development Best Practice**
>
> Test in SWI-Prolog first! It has better error messages and debugging tools than GNU Prolog.

### Step 3: Generate GNU Prolog Script

Use UnifyWeaver's Prolog target to generate a GNU Prolog script:

```bash
cd $UNIFYWEAVER_HOME

swipl -q <<'PROLOG_SCRIPT'
% Load Prolog target
asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
use_module(unifyweaver(targets/prolog_target)),

% Load our factorial predicate
['/tmp/factorial_gnu.pl'],

% Generate GNU Prolog script with compilation enabled
generate_prolog_script(
    [factorial/2, test_factorial/0],
    [
        dialect(gnu),
        compile(true),
        entry_point(test_factorial)
    ],
    Code
),

% Write script with automatic compilation
write_prolog_script(
    Code,
    'factorial_gnu',
    [dialect(gnu), compile(true)]
),

format('~n✓ Generation complete~n'),
halt.
PROLOG_SCRIPT
```

**Expected output**:
```
[PrologTarget] Generated script: factorial_gnu.pl
[PrologTarget] Compiling with gnu: gplc --no-top-level factorial_gnu.pl -o factorial_gnu
[PrologTarget] Compilation complete

✓ Generation complete
```

**Verification**: Both script and binary should exist
```bash
ls -lh factorial_gnu factorial_gnu.pl
# Should show both files
```

> [!note]
> The `.pl` extension is added automatically to the script. The binary has no extension.

### Step 4: Inspect Generated Script

Examine the generated GNU Prolog code:

```bash
head -n 50 factorial_gnu.pl
```

**Key elements to look for**:
1. **Shebang**: `#!/usr/bin/env gprolog --consult-file`
2. **Header comments**: UnifyWeaver version, timestamp, target info
3. **User predicates**: Your factorial/2 and test_factorial/0
4. **main/0 predicate**: Entry point with error handling
5. **Initialization**: `:- initialization(test_factorial).` for compiled mode

> [!tip]
> **Understanding Initialization**
>
> For compiled binaries (`gplc --no-top-level`), GNU Prolog requires:
> ```prolog
> :- initialization(Goal).
> ```
> NOT just `:- Goal.` (which works for interpreted mode).
>
> UnifyWeaver automatically generates the correct directive based on the `compile(true)` option.

### Step 5: Execute the Compiled Binary

Run the standalone binary:

```bash
./factorial_gnu
```

**Expected output**:
```
Factorial of 5 is 120
Factorial of 10 is 3628800
```

**Verification**: Check exit code
```bash
./factorial_gnu && echo "Exit code: $?"
# Expected: Exit code: 0
```

> [!tip]
> **Fast Startup**
>
> Notice how fast the binary starts! No interpreter loading.
> ```bash
> time ./factorial_gnu
> # Typically <5ms on modern hardware
> ```

### Step 6: Compare with Interpreted Mode

For comparison, run the generated script in interpreted mode:

```bash
gprolog --consult-file factorial_gnu.pl
```

**Expected**: Same output as binary, but slower startup.

### Step 7: Verify Portability

The binary is portable within the same architecture:

```bash
file factorial_gnu
# Example: factorial_gnu: ELF 64-bit LSB executable, x86-64

ldd factorial_gnu
# Shows dynamic library dependencies (or "not a dynamic executable" if statically linked)
```

## Expected Outputs

> [!output]
> language: prolog
> purpose: GNU Prolog script with correct initialization
> format: executable_script
> location: factorial_gnu.pl
>
> ```prolog
> #!/usr/bin/env gprolog --consult-file
>
> % Generated by UnifyWeaver v0.1
> % Target: Prolog (GNU Prolog)
> % Compilation: interpreted
> % Generated: 2025-11-17 22:00:00
>
> % === User Code (Transpiled) ===
>
> factorial(0, 1) :- !.
> factorial(A, B) :-
>     A>0,
>     C is A+ -1,
>     factorial(C, D),
>     B is A*D.
>
> test_factorial :-
>     factorial(5, A),
>     format('Factorial of 5 is ~w~n', [A]),
>     factorial(10, B),
>     format('Factorial of 10 is ~w~n', [B]).
>
> % === Entry Point ===
>
> main :-
>     test_factorial,
>     halt(0).
>
> main :-
>     format(user_error, 'Error: Execution failed~n', []),
>     halt(1).
>
> % Entry point (for compiled binary)
> :- initialization(test_factorial).
> ```

> [!output]
> language: binary
> purpose: Native GNU Prolog executable
> format: executable_binary
> location: factorial_gnu
> expected_size: ~800KB-2MB

## Verification

### Binary Execution Test

```bash
# Test 1: Run binary
./factorial_gnu
# Expected: Output with factorials, exit 0

# Test 2: Check it's truly a binary (not a script)
file factorial_gnu | grep -q "executable" && echo "✓ Is executable binary"

# Test 3: Measure startup time
time ./factorial_gnu > /dev/null
# Expected: Real time <10ms
```

### Interpreted Mode Test

```bash
# Test with interpreter (for comparison)
time gprolog --consult-file factorial_gnu.pl > /dev/null
# Expected: Real time ~50-100ms (slower startup)
```

### Success Criteria

✅ All of the following must be true:
- `/tmp/factorial_gnu.pl` exists with valid Prolog code
- `factorial_gnu.pl` generated in current directory
- `factorial_gnu` binary exists and is executable
- Binary outputs correct factorials (120 and 3628800)
- Binary exits with code 0
- Binary startup time <10ms
- Generated script has `:- initialization(test_factorial).` directive
- Interpreted mode also works with the same script

## Troubleshooting

### Error: "gplc: command not found"

**Cause**: GNU Prolog compiler not installed

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install gprolog

# macOS
brew install gnu-prolog

# Verify installation
gplc --version
```

### Error: "Warning: no initial goal executed"

**Cause**: Wrong initialization directive in generated code

**Solution**: This should not happen with UnifyWeaver v0.1+, which automatically generates `:- initialization(Goal).` for compiled mode. If you see this error:

1. Check the generated script:
   ```bash
   grep "initialization" factorial_gnu.pl
   # Should show: :- initialization(test_factorial).
   ```

2. If it shows `:- test_factorial.` instead, you may be using an old version:
   ```bash
   cd $UNIFYWEAVER_HOME
   git pull origin main
   ```

### Error: "Compilation failed with exit code 1"

**Cause**: Syntax error in generated Prolog code

**Solution**:
1. Test the generated script in interpreted mode first:
   ```bash
   gprolog --consult-file factorial_gnu.pl
   ```
2. If it fails, check for syntax errors:
   ```bash
   gprolog < factorial_gnu.pl
   ```
3. Common issues:
   - Missing periods at end of clauses
   - Unmatched parentheses
   - Invalid operators

### Binary runs but produces no output

**Cause**: Entry point not called or output buffering

**Solution**:
1. Verify main/0 is called:
   ```bash
   gprolog --consult-file factorial_gnu.pl -g "main"
   ```
2. Check initialization directive exists:
   ```bash
   tail -5 factorial_gnu.pl
   # Should end with: :- initialization(test_factorial).
   ```

### Graceful Fallback (Compilation Fails)

If compilation fails, UnifyWeaver logs a warning but continues with the interpreted script:

```
[PrologTarget] WARNING: gnu compilation failed (exit 1)
[PrologTarget] Continuing with interpreted script: factorial_gnu.pl
```

**What to do**: You can still use the script in interpreted mode:
```bash
gprolog --consult-file factorial_gnu.pl
```

For strict mode (fail on compilation errors):
```bash
# Add fail_on_compile_error(true) option
write_prolog_script(Code, 'factorial_gnu', [
    dialect(gnu),
    compile(true),
    fail_on_compile_error(true)
])
```

## References

- **[Book: Prolog Target](../../book-prolog-target/README.md)** - Complete guide to Prolog target
  - [Chapter 3: Dialects](../../book-prolog-target/03_dialects.md) - SWI vs GNU Prolog
  - [Chapter 5: Compilation Modes](../../book-prolog-target/05_compilation_modes.md) - Compiled vs interpreted
  - [Chapter 15: Case Study](../../book-prolog-target/15_case_study.md) - Factorial example walkthrough

- **Example Libraries**:
  - [Recursion Examples](../examples_library/recursion_examples.md)
  - [Compilation Examples](../examples_library/compilation_examples.md)

- **UnifyWeaver Documentation**:
  - [Prolog Target API](../../book-prolog-target/appendix_a_api_reference.md)
  - [Firewall Integration](../../book-prolog-target/07_firewall_integration.md)
  - [Fallback Mechanisms](../../book-prolog-target/08_fallback_mechanisms.md)

## Next Steps

After completing this playbook:

1. **Explore Dialect Features**:
   - Read [Chapter 3: Dialects](../../book-prolog-target/03_dialects.md) to understand SWI vs GNU differences
   - Try compiling predicates with different GNU Prolog features

2. **Static Linking** (for truly portable binaries):
   ```bash
   gplc --no-top-level --static factorial_gnu.pl -o factorial_gnu_static
   # Creates binary with no external dependencies
   ```

3. **Multi-Target Deployment**:
   - Generate both SWI and GNU versions
   - Compare performance characteristics
   - Deploy appropriate version per platform

4. **Production Workflow**:
   - Develop with SWI-Prolog (rich tooling)
   - Test with validation (ensure compatibility)
   - Deploy as GNU Prolog binary (fast, standalone)

## Notes

> [!tip]
> **Development Workflow**
>
> Best practice for production deployments:
> 1. Develop and debug in SWI-Prolog (better tools)
> 2. Validate with `validate_for_dialect(gnu, [your_pred/arity], Issues)`
> 3. Generate and test GNU Prolog interpreted version
> 4. Compile to binary for final deployment
> 5. Test binary on target platform

> [!warning]
> **Binary Compatibility**
>
> Compiled binaries are platform-specific:
> - x86_64 Linux binary won't run on ARM or macOS
> - Recompile on each target platform
> - Or use static linking for better portability

> [!tip]
> **When Compilation is Worth It**
>
> Use compiled binaries when:
> - Deployment to end users (no Prolog installation needed)
> - Fast startup is critical (command-line tools)
> - Want single-file distribution
> - Running on resource-constrained systems
>
> Stick with interpreted when:
> - Rapid development and testing
> - Debugging needed
> - Code changes frequently
> - Using SWI-specific features (HTTP, databases, etc.)

> [!note]
> **The v0.1 Improvements**
>
> The Prolog target in UnifyWeaver v0.1 includes critical fixes:
> 1. ✅ Correct initialization directives for compiled binaries
> 2. ✅ Exit code checking with error detection
> 3. ✅ Graceful fallback when compilation fails
>
> See [Chapter 15: Case Study](../../book-prolog-target/15_case_study.md) for the full story of how these were discovered and fixed.

---

**Playbook Version**: 1.0
**Last Updated**: 2025-11-17
**Tested With**: UnifyWeaver v0.1, SWI-Prolog 8.x, GNU Prolog 1.5.0

**Execution Time**: ~10 minutes
**Success Rate**: 95%+ with correct prerequisites
**Platform**: Linux, macOS (with GNU Prolog), WSL
