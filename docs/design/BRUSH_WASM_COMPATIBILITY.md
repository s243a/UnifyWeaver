# Brush WASM Compatibility Notes

UnifyWeaver's SciREPL package runs generated Bash scripts inside
[Brush](https://github.com/reubeno/brush), a Rust-based bash-compatible
shell compiled to WebAssembly. Brush aims for full bash compatibility but
runs on a single-threaded WASM runtime without OS process support. This
creates a small set of behavioural differences that affect generated code.

Note: these limitations are **not Brush-specific** — they stem from the
`wasm32-unknown-unknown` target, which provides no POSIX layer at all.
There are no syscalls for process management (`getpid`, `fork`), signals
(`SIGPIPE`), or filesystem access (`open`, `stat`). All I/O is bridged
through JavaScript via `wasm-bindgen`. A WASI target
(`wasm32-wasip1/p2`) would provide some of these syscalls natively, but
SciREPL uses `wasm32-unknown-unknown` + JS VFS interop because the
notebook's SharedVFS architecture requires tight JS integration for
cross-kernel file sharing.

## Known Differences

### 1. `$$` (Process ID) hangs

**Symptom:** Any script referencing `$$` hangs indefinitely.

**Cause:** `std::process::id()` calls `getpid()`, which is not available
on `wasm32-unknown-unknown`. The call blocks forever instead of returning
an error.

**Fix applied:** Return a dummy PID (`1`) on WASM
(`brush-core/src/expansion.rs`, `SpecialParameter::ProcessId`).

**Status:** Fixed in our fork. Not yet upstreamed.

### 2. `trap ... PIPE` — invalid signal

**Symptom:** `trap "cleanup" EXIT PIPE` errors with
`trap: PIPE: invalid signal specification`.

**Cause:** SIGPIPE doesn't exist on WASM (no OS signals). Brush's signal
table doesn't include it.

**Workaround applied:** UnifyWeaver's transitive closure template uses
`trap ... EXIT` without `PIPE`. The PIPE trap was only a best-effort
SIGPIPE guard for the `tee >(grep -q ...)` pattern; EXIT alone handles
cleanup correctly.

**Status:** Workaround in UnifyWeaver templates. Could also be patched in
Brush to silently ignore unknown signals on WASM instead of erroring.

### 3. `tee >(cmd)` — output process substitution deadlocks

**Symptom:** `echo x | tee >(grep y) >/dev/null` hangs forever.

**Cause:** On native platforms, Brush uses `tokio::spawn` to run the
`>(cmd)` subshell concurrently with the parent. On single-threaded WASM
tokio, the spawned task only gets polled when the parent yields. If the
parent (e.g. `tee`) writes synchronously without yielding, the subshell
never runs, and the shell deadlocks.

**Fix applied:** On WASM, Write process substitution uses a temp-file +
deferred execution approach: the parent writes to a real temp file
(not `/dev/fd/NN`), and after the parent command finishes, the subshell
runs synchronously with the temp file as stdin
(`brush-core/src/interp.rs`).

**Note:** The temp file path is passed directly in argv (not via
`/dev/fd/NN`) because uutils builtins like `tee` call
`wasm_open_file()` which bypasses the `/dev/fd/NN` resolver in
`open_file_with_mode()`.

**Status:** Fixed in our fork. Candidate for upstream contribution.

### 4. `[ -f file ]`, `[ -e file ]`, etc. — false negatives for VFS files

**Symptom:** `touch /tmp/foo && [ -f /tmp/foo ]` returns false.

**Cause:** Test predicates (`-f`, `-e`, `-d`, `-r`, `-s`) used
`std::path::Path` methods which hit the real filesystem and ignore the
JS-backed SharedVFS where Brush WASM stores files.

**Fix applied:** On WASM, check `wasm_file_exists()` / `wasm_stat()`
first, then fall back to the real filesystem
(`brush-core/src/extendedtests.rs`).

**Status:** Fixed in our fork. Should be upstreamed.

### 5. `ls file` — "No such file or directory" for single files

**Symptom:** `ls /tmp/` works but `ls /tmp/foo` fails even when the file
exists.

**Cause:** The `ls` builtin only called `wasm_list_dir()`, which returns
`None` for non-directories.

**Fix applied:** Fall back to `wasm_stat()` for single-file listings
(`brush-uutils/src/lib.rs`).

**Status:** Fixed in our fork. Should be upstreamed.

### 6. `mv` — not implemented

**Symptom:** `mv src dst` errors with
`failed to execute command 'mv': operation not supported on this platform`.

**Cause:** No `mv` builtin existed in brush-uutils.

**Fix applied:** Added `MvBuiltin` that reads source via VFS, writes to
destination, removes source (`brush-uutils/src/lib.rs`).

**Status:** Fixed in our fork. Should be upstreamed.

## Design Decision: How to Handle Incompatibilities

We considered three approaches and chose option 2.

### Option 1: Brush Target

Add a dedicated `brush` compilation target to UnifyWeaver that generates
Brush-compatible bash — avoiding `trap PIPE`, `$$` in filenames, output
process substitution, and any other known gaps.

**Pros:**
- Clean separation; generated code is guaranteed to work on Brush.
- No need to patch Brush for every gap.

**Cons:**
- Another target to maintain (templates, tests, documentation).
- Brush aims for bash compatibility, so the gap should shrink over time,
  making a separate target increasingly redundant.
- Risk of the two targets drifting apart, with bugs fixed in one but not
  the other.

### Option 2: Workaround in Default Bash Templates (chosen)

Adjust the default bash templates to avoid constructs that are
problematic on Brush, as long as the change is also valid (or harmless)
on real bash.

**Pros:**
- One target, one set of templates.
- Changes like `trap EXIT` instead of `trap EXIT PIPE` are correct on
  all platforms.
- Minimal maintenance burden.

**Cons:**
- Only works when the workaround is platform-neutral. If a future
  incompatibility requires Brush-specific code that would break real
  bash, this approach fails and we'd need option 1 or 3.

### Option 3: Patch Brush

Fix each incompatibility directly in the Brush source (our fork) and
rebuild the WASM binary.

**Pros:**
- Generated code stays idiomatic bash; no workarounds needed.
- Benefits all Brush users if upstreamed.

**Cons:**
- Requires Rust + wasm-pack toolchain for every fix.
- Rebuild cycle is ~90 seconds; adds friction during development.
- Some gaps (like SIGPIPE) are inherent to WASM's lack of OS signals
  and may not have clean Brush-level fixes.

### Predicate-Controlled Behaviour

Regardless of which option is the default, users should be able to opt
into or out of Brush-specific workarounds via a constraint or option
predicate. For example:

```prolog
% Use Brush-compatible workarounds (the default in SciREPL context)
:- set_constraint(ancestor/2, shell_compat(brush)).

% Use full bash features (for native execution)
:- set_constraint(ancestor/2, shell_compat(bash)).

% Or as a compile option:
compile_recursive(ancestor/2, [shell(brush)], Code).
```

When `shell(brush)` is active, the compiler would:
- Use `trap ... EXIT` without `PIPE`
- Avoid `$$` in temp file names (use `$RANDOM` or a counter instead)
- Avoid output process substitution `tee >(cmd)` in favour of temp
  files or sequential pipelines
- Prefer builtins known to exist in brush-uutils

When `shell(bash)` is active (or unspecified on native), the compiler
generates idiomatic bash with no restrictions.

This keeps option 2 as the default while giving users an escape hatch
in both directions. The predicate could also be set globally via
`preferences.pl` so SciREPL workbooks automatically compile with
Brush compatibility without per-predicate annotations.

### Recommendation

Use option 2 as the default. When a workaround would compromise the
generated code's correctness or readability on real bash, fall back to
option 3 (patch Brush). Reserve option 1 for if the gap grows large
enough to justify a dedicated target — currently it does not.

The `shell_compat` predicate (described above) provides a migration path:
if option 1 becomes necessary in the future, the predicate-based
switching is already in place.

## Brush Fork Location

Our patched Brush lives at `~/Projects/brush` (local) and
`github.com/s243a/brush` (remote). The WASM binary is committed to the
SciREPL submodule at
`examples/sci-repl/prototype/www/vendor/brush/brush_wasm_bg.wasm`.

## Upstream Contribution Candidates

The following fixes are general improvements that would benefit all Brush
WASM users and are good candidates for upstream PRs to
`github.com/reubeno/brush`:

1. `$$` returning dummy PID on WASM
2. Write process substitution temp-file approach on WASM
3. Test predicates checking VFS before real filesystem
4. `ls` single-file fallback to `wasm_stat`
5. `mv` builtin for WASM
