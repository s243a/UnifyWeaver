# Chapter 8: Dialect Fallback Mechanisms

## Introduction to Fallback

One of the most powerful features of the Prolog target is its ability to gracefully handle failures and automatically try alternative approaches. This chapter explores the fallback mechanisms that make the Prolog target robust and production-ready.

## Why Fallback Matters

Consider these real-world scenarios:

1. **Compiler not available**: User requests GNU Prolog compilation, but `gplc` isn't installed
2. **Compilation failure**: Code compiles in SWI-Prolog but fails with GNU Prolog
3. **Platform limitations**: Target platform doesn't support binary execution
4. **Firewall restrictions**: Security policy blocks the preferred dialect

Without fallback, these would be hard failures. With fallback, UnifyWeaver can:
- Try alternative dialects
- Fall back to interpreted mode
- Continue with best-effort execution
- Provide helpful diagnostics

## Types of Fallback

### 1. Compilation Failure Fallback

**Scenario**: Requested compilation fails

**Implementation**: `compile_script_safe/3`

```prolog
compile_script_safe(Dialect, ScriptPath, Options) :-
    catch(
        compile_script(Dialect, ScriptPath),
        error(compilation_failed(FailedDialect, ExitCode), Context),
        (   % Compilation failed - handle gracefully
            format('[PrologTarget] WARNING: ~w compilation failed (exit ~w)~n',
                   [FailedDialect, ExitCode]),

            % Check if we should fail or continue
            (   option(fail_on_compile_error(true), Options)
            ->  % Propagate error if strict mode
                throw(error(compilation_failed(FailedDialect, ExitCode), Context))
            ;   % Otherwise continue with interpreted script
                format('[PrologTarget] Continuing with interpreted script: ~w~n', [ScriptPath]),
                format('[PrologTarget] NOTE: Full multi-dialect fallback planned for v0.2~n')
            )
        )
    ).
```

**Behavior**:
- Attempts compilation with `gplc`
- Catches non-zero exit codes
- Logs warning
- **Default**: Continue with interpreted script
- **Strict mode** (`fail_on_compile_error(true)`): Re-throw error

**Example**:

```prolog
% Request compilation
write_prolog_script(Code, 'test.pl', [dialect(gnu), compile(true)]).

% If gplc fails:
% [PrologTarget] WARNING: gnu compilation failed (exit 1)
% [PrologTarget] Continuing with interpreted script: test.pl
% [PrologTarget] NOTE: Full multi-dialect fallback planned for v0.2

% Script still created and usable with: gprolog --consult-file test.pl
```

### 2. Preference-Based Fallback

**Scenario**: Multiple dialects specified in preference order

**Current Status**: Planned for v0.2

**Proposed API**:

```prolog
generate_prolog_script(Predicates,
                      [dialects([gnu, swi]),  % Preference order
                       compile(true),
                       fallback(auto)],
                      Code)
```

**Planned Behavior**:

1. Try GNU Prolog compilation
2. If compilation fails, try GNU Prolog interpreted
3. If that fails (e.g., compatibility issues), regenerate for SWI-Prolog
4. If that fails, report all failures

**Implementation Sketch**:

```prolog
generate_with_fallback([Dialect|Rest], Predicates, Options, Code) :-
    catch(
        (   % Try current dialect
            generate_prolog_script_single(Predicates,
                                          [dialect(Dialect)|Options],
                                          Code),
            % Validate if compilation requested
            (   option(compile(true), Options)
            ->  validate_compilation(Dialect, Code)
            ;   true
            )
        ),
        Error,
        (   % Failed - try next dialect
            format('[Fallback] ~w failed: ~w~n', [Dialect, Error]),
            Rest \= []
        ->  generate_with_fallback(Rest, Predicates, Options, Code)
        ;   % No more fallbacks
            throw(Error)
        )
    ).
```

### 3. Firewall-Based Fallback

**Scenario**: Security policy restricts dialect usage

**Integration**: Firewall policies can block certain dialects

```prolog
% In firewall policy
:- disallow(prolog_target(gnu)).
:- disallow(compile(native)).
```

**Behavior**:

```prolog
% User requests GNU Prolog
generate_prolog_script(Predicates, [dialect(gnu)], Code).

% Firewall blocks it:
% [Firewall] DENIED: prolog_target(gnu) - policy violation
% [Fallback] Trying alternative: swi

% Falls back to SWI-Prolog
```

**Implementation**:

```prolog
select_dialect_with_policy(RequestedDialect, Options, SelectedDialect) :-
    % Check firewall
    (   firewall_allows(prolog_target(RequestedDialect))
    ->  SelectedDialect = RequestedDialect
    ;   % Blocked - find allowed alternative
        format('[Firewall] DENIED: prolog_target(~w)~n', [RequestedDialect]),
        option(fallback_dialects(Alternatives), Options, [swi]),
        member(AltDialect, Alternatives),
        firewall_allows(prolog_target(AltDialect)),
        !,
        format('[Fallback] Using allowed dialect: ~w~n', [AltDialect]),
        SelectedDialect = AltDialect
    ).
```

### 4. Capability-Based Fallback

**Scenario**: Code uses features not available in requested dialect

**Example**: Code uses SWI-specific features but requests GNU Prolog

```prolog
% Code uses tabling (SWI-Prolog specific)
:- table fib/2.
fib(0, 0) :- !.
fib(1, 1) :- !.
fib(N, F) :- N > 1, N1 is N-1, N2 is N-2, fib(N1, F1), fib(N2, F2), F is F1+F2.
```

**Validation**:

```prolog
validate_for_dialect(gnu, [fib/2], Issues).
% Issues = [unsupported_feature(tabling, 'GNU Prolog does not support tabling')]
```

**Fallback**:

```prolog
generate_with_validation(Predicates, Options, Code) :-
    option(dialect(RequestedDialect), Options, swi),

    % Validate
    validate_for_dialect(RequestedDialect, Predicates, Issues),

    (   Issues = []
    ->  % No issues - proceed
        generate_prolog_script(Predicates, Options, Code)
    ;   % Has issues - check fallback
        format('[Validation] Issues with ~w: ~w~n', [RequestedDialect, Issues]),
        (   option(fallback(auto), Options)
        ->  % Try alternative
            find_compatible_dialect(Predicates, CompatibleDialect),
            format('[Fallback] Using compatible dialect: ~w~n', [CompatibleDialect]),
            generate_prolog_script(Predicates,
                                  [dialect(CompatibleDialect)|Options],
                                  Code)
        ;   % No fallback - fail with issues
            throw(error(incompatible_dialect(RequestedDialect, Issues)))
        )
    ).
```

## Fallback Strategies

### Strategy 1: Silent Fallback (Current Default)

**When**: Compilation fails
**Action**: Continue with interpreted script, log warning
**Use case**: Development, testing

```prolog
Options = [dialect(gnu), compile(true)]
% If compilation fails, script is still usable interpreted
```

### Strategy 2: Strict Mode

**When**: Any failure
**Action**: Throw error immediately
**Use case**: CI/CD, production validation

```prolog
Options = [dialect(gnu), compile(true), fail_on_compile_error(true)]
% Compilation failure → error thrown
```

### Strategy 3: Multi-Dialect Fallback (Planned v0.2)

**When**: Requested dialect fails
**Action**: Try alternatives in order
**Use case**: Maximum portability

```prolog
Options = [dialects([gnu, swi]), fallback(auto), compile(try)]
% Try: GNU compiled → GNU interpreted → SWI interpreted
```

### Strategy 4: Best-Effort (Planned v0.2)

**When**: Target has limitations
**Action**: Generate code with workarounds
**Use case**: Feature-limited environments

```prolog
Options = [dialect(gnu), fallback(best_effort)]
% Remove unsupported features, add comments explaining changes
```

## Fallback Decision Tree

```
User Request
    │
    ├─ Check Firewall
    │   ├─ Allowed → Continue
    │   └─ Blocked → Try alternatives or fail
    │
    ├─ Validate Compatibility
    │   ├─ Compatible → Continue
    │   └─ Incompatible → Fallback or fail
    │
    ├─ Generate Code
    │   ├─ Success → Continue
    │   └─ Failure → Fallback or fail
    │
    └─ Compile (if requested)
        ├─ Success → Done ✓
        ├─ Failure + Silent → Use interpreted ✓
        └─ Failure + Strict → Error ✗
```

## Implementing Custom Fallback

You can implement custom fallback logic:

```prolog
my_robust_generate(Predicates, PreferredDialect, Code) :-
    % Strategy: Try preferred, then any compatible, then fail with diagnostics

    catch(
        % Attempt 1: Preferred dialect
        generate_prolog_script(Predicates,
                              [dialect(PreferredDialect), compile(true)],
                              Code),
        Error1,
        (   format('[Attempt 1] ~w failed: ~w~n', [PreferredDialect, Error1]),

            % Attempt 2: Find compatible dialect
            catch(
                (   find_compatible_dialect(Predicates, AltDialect),
                    AltDialect \= PreferredDialect,
                    format('[Attempt 2] Trying ~w~n', [AltDialect]),
                    generate_prolog_script(Predicates,
                                          [dialect(AltDialect), compile(false)],
                                          Code)
                ),
                Error2,
                (   format('[Attempt 2] ~w failed: ~w~n', [AltDialect, Error2]),

                    % Attempt 3: SWI-Prolog interpreted (most compatible)
                    format('[Attempt 3] Falling back to SWI-Prolog interpreted~n'),
                    generate_prolog_script(Predicates,
                                          [dialect(swi), compile(false)],
                                          Code)
                )
            )
        )
    ).
```

## Logging and Diagnostics

All fallback attempts are logged for debugging:

```prolog
% Enable verbose logging
set_prolog_flag(verbose, normal).

% Generate with logging
generate_prolog_script([factorial/2],
                      [dialect(gnu), compile(true), verbose(true)],
                      Code).

% Output:
% [PrologTarget] Dialect: gnu
% [PrologTarget] Compilation requested: true
% [PrologTarget] Generating script...
% [PrologTarget] Compiling with gnu: gplc --no-top-level ...
% [PrologTarget] Compilation complete
```

## Roadmap: Future Fallback Features (v0.2)

### Planned Enhancements

1. **Multi-Dialect Chains**:
   ```prolog
   dialects([gnu(compiled), gnu(interpreted), swi(interpreted)])
   ```

2. **Feature Degradation**:
   ```prolog
   fallback(degrade)  % Remove unsupported features, continue
   ```

3. **Automatic Dialect Selection**:
   ```prolog
   dialect(auto)  % Analyze code, pick best dialect
   ```

4. **Compilation Caching**:
   ```prolog
   compile(cached)  % Reuse successful compilations
   ```

5. **Fallback Hooks**:
   ```prolog
   on_fallback(log_and_notify)  % Custom actions on fallback
   ```

## Best Practices

### For Users

1. **Specify preferences explicitly**:
   ```prolog
   [dialects([gnu, swi]), compile(true), fallback(auto)]
   ```

2. **Use strict mode in CI/CD**:
   ```prolog
   [fail_on_compile_error(true)]
   ```

3. **Test with multiple dialects**:
   ```prolog
   % Test both paths
   generate_prolog_script(Preds, [dialect(gnu)], GnuCode),
   generate_prolog_script(Preds, [dialect(swi)], SwiCode).
   ```

### For Implementers

1. **Catch specific errors**:
   ```prolog
   catch(..., error(compilation_failed(D, E), C), ...)
   ```

2. **Log all fallback attempts**:
   ```prolog
   format('[Fallback] Trying ~w after ~w failure~n', [Alt, Orig])
   ```

3. **Provide diagnostics**:
   ```prolog
   throw(error(all_dialects_failed(Attempts), Context))
   ```

## Example: Complete Fallback Chain

```prolog
robust_compile(Predicates, OutputPath) :-
    % Try GNU compiled
    catch(
        (   generate_prolog_script(Predicates,
                                  [dialect(gnu), compile(true)],
                                  Code1),
            write_prolog_script(Code1, OutputPath,
                               [dialect(gnu), compile(true)])
        ),
        _,
        % Fall back to GNU interpreted
        catch(
            (   generate_prolog_script(Predicates,
                                      [dialect(gnu), compile(false)],
                                      Code2),
                write_prolog_script(Code2, OutputPath,
                                   [dialect(gnu), compile(false)])
            ),
            _,
            % Fall back to SWI interpreted
            (   generate_prolog_script(Predicates,
                                      [dialect(swi), compile(false)],
                                      Code3),
                write_prolog_script(Code3, OutputPath,
                                   [dialect(swi), compile(false)])
            )
        )
    ).
```

## What's Next?

Chapter 9 explores validation and compatibility checking in depth, showing how to detect issues before they cause fallback situations.

---

**Key Takeaways:**
- Fallback mechanisms provide robustness and flexibility
- Current implementation handles compilation failures gracefully
- v0.2 will add multi-dialect fallback chains
- Firewall integration enables security-driven fallback
- Proper logging aids debugging and diagnostics
- Balance between automatic fallback and strict validation

---

## Navigation

**←** [Previous: Chapter 7: The Firewall System](07_firewall_integration) | [📖 Book 11: Prolog Target](README) | [Next: Chapter 15: Case Study - Factorial Compilation →](15_case_study)
