# Handoff Back to Gemini — XML Source Test Fix

**Date:** 2025-10-30  
**From:** cline.bot  
**To:** Gemini

---

## 1. Summary

- Targeted issue: `read_util:read_stream_to_codes/2` was being called with uninstantiated stream arguments inside `check_lxml_available/0`, triggered during `xml_source:compile_source/4` when the lxml availability check failed.  
- Resolution: rewrote the availability check to iterate over multiple Python interpreter candidates, spawning each within a `setup_call_cleanup/3` block so that stream handles are only read if process creation succeeds, and are always closed afterward. Exceptions from missing executables are ignored; other errors are reported and cause failure.  
- Verification: WSL-based Prolog test `tests/core/test_xml_source.pl` (`run_tests/0`) now passes (`basic_extraction` succeeds, only original choicepoint warning remains). No other files required changes.

---

## 2. Root Cause

The original implementation of `check_lxml_available/0` hard-coded a call to `C:\Windows\System32\wsl.exe` and wrapped it in a `catch/3` that, on any exception, still attempted to read the `Out` and `Err` pipes. When the executable could not be spawned (e.g., path mismatch on some systems), the stream variables never instantiated; the subsequent `read_stream_to_codes/2` calls raised the instantiation error observed in PLUnit.

---

## 3. Changes Made

| File | Description |
| ---- | ----------- |
| `src/unifyweaver/sources/xml_source.pl` | Added `library(process)` and `library(readutil)` imports. Replaced `check_lxml_available/0` with a candidate-driven implementation that safely manages process handles and handles missing interpreters gracefully. |

---

## 4. Key Code Snippet

Updated `check_lxml_available/0`:

```prolog
check_lxml_available :-
    python_lxml_candidates(Candidates),
    member(Exec-Args, Candidates),
    python_lxml_check(Exec, Args),
    !.

check_lxml_available :-
    format('lxml check failed using all configured interpreters.~n', []),
    fail.

python_lxml_candidates([
    path(python3)-['-c', 'import lxml'],
    path(py)-['-3', '-c', 'import lxml'],
    path(wsl)-['python3', '-c', 'import lxml'],
    path('C:\\Windows\\System32\\wsl.exe')-['python3', '-c', 'import lxml']
]).

python_lxml_check(Exec, Args) :-
    catch(
        setup_call_cleanup(
            process_create(Exec, Args, [stdout(pipe(Out)), stderr(pipe(Err)), process(Process)]),
            (
                read_stream_to_codes(Out, OutCodes),
                read_stream_to_codes(Err, ErrCodes),
                process_wait(Process, ExitStatus),
                (   ExitStatus = exit(0)
                ->  true
                ;   format('lxml check via ~q failed (status ~w). stdout: ~s, stderr: ~s~n',
                           [Exec, ExitStatus, OutCodes, ErrCodes]),
                    fail
                )
            ),
            (   close(Out),
                close(Err)
            )
        ),
        Error,
        (   (   Error = error(existence_error(_, _), _)
            ->  fail                 % interpreter not present — try next candidate
            ;   format('lxml check via ~q raised exception: ~q~n', [Exec, Error]),
                fail
            )
        )
    ).
```

This approach ensures `read_stream_to_codes/2` is only called when the stream handles exist, and any missing interpreter simply causes iteration to advance to the next candidate.

---

## 5. Tests Executed

```
wsl swipl -s tests/core/test_xml_source.pl -g run_tests -t halt
```

Result: `basic_extraction` passes; the runner emits the existing choicepoint warning but no errors.

---

## 6. Files Modified

- `src/unifyweaver/sources/xml_source.pl`

_No other files touched._

---

## 7. Next Steps / Recommendations

- If future platforms require different Python entry points, extend `python_lxml_candidates/1` accordingly.  
- Implement the `xmllint` fallback engine when ready; the current fix ensures the test suite no longer errors out before reaching that branch.

This document should give you all context needed to understand the applied fix and resume work.
