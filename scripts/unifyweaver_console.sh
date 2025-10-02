#!/usr/bin/env bash
# UnifyWeaver Console Launcher
# Supports Cygwin, MSYS2, WSL, and native Linux
set -eu
set -o pipefail 2>/dev/null || true

if [[ -z "${UNIFYWEAVER_ROOT:-}" ]]; then
  SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  UNIFYWEAVER_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
fi

echo "[INFO] UNIFYWEAVER_ROOT: $UNIFYWEAVER_ROOT"
cd "$UNIFYWEAVER_ROOT"
echo "[INFO] pwd: $(pwd)"

if ! command -v swipl >/dev/null 2>&1; then
  echo "[ERROR] swipl not found on PATH." >&2
  echo "[ERROR] Please install SWI-Prolog:" >&2
  echo "  Windows: https://www.swi-prolog.org/download/stable" >&2
  echo "  Linux:   sudo apt install swi-prolog" >&2
  exit 1
fi

# Convert to Windows path format and fix backslashes for Prolog (Cygwin/MSYS2)
if command -v cygpath >/dev/null 2>&1; then
  WIN_UNIFYWEAVER_ROOT="$(cygpath -w "$UNIFYWEAVER_ROOT" | sed 's|\\|/|g')"
  echo "[INFO] Windows path (Prolog-safe): $WIN_UNIFYWEAVER_ROOT"
else
  WIN_UNIFYWEAVER_ROOT="$UNIFYWEAVER_ROOT"
fi

# Define initialization goal
# Sets up library paths and defines helper predicates
read -r -d '' PROLOG_GOAL <<PL || true
( working_directory(CWD, CWD),
  format('[UnifyWeaver] Working directory: ~w~n', [CWD]),
  atom_string('$WIN_UNIFYWEAVER_ROOT', UnifyweaverRootStr),
  atom_concat(UnifyweaverRootStr, '/src', AbsSrcDir),
  atom_concat(UnifyweaverRootStr, '/src/unifyweaver', AbsUnifyweaverDir),
  asserta(user:library_directory(AbsSrcDir)),
  asserta(file_search_path(unifyweaver, AbsUnifyweaverDir)),
  format('[UnifyWeaver] Absolute paths configured:~n', []),
  format('  src: ~w~n', [AbsSrcDir]),
  format('  unifyweaver: ~w~n', [AbsUnifyweaverDir]),
  format('~n[UnifyWeaver] Console ready.~n', []),
  format('Available commands:~n', []),
  format('  load_recursive.  - Load recursive_compiler~n', []),
  format('  load_stream.     - Load stream_compiler~n', []),
  format('  load_template.   - Load template_system~n', []),
  format('  load_all_core.   - Load all core modules~n', []),
  format('  test_advanced.   - Test advanced recursion~n', []),
  format('  help.            - Show help~n~n', []),

  % Define helper predicates
  asserta((load_recursive :-
    (use_module(unifyweaver(core/recursive_compiler)) ->
      format('recursive_compiler loaded successfully!~n', [])
    ; format('Failed to load recursive_compiler~n', [])))),

  asserta((load_stream :-
    (use_module(unifyweaver(core/stream_compiler)) ->
      format('stream_compiler loaded successfully!~n', [])
    ; format('Failed to load stream_compiler~n', [])))),

  asserta((load_template :-
    (use_module(unifyweaver(core/template_system)) ->
      format('template_system loaded successfully!~n', [])
    ; format('Failed to load template_system~n', [])))),

  asserta((load_all_core :-
    use_module(unifyweaver(core/recursive_compiler)),
    use_module(unifyweaver(core/stream_compiler)),
    use_module(unifyweaver(core/template_system)),
    format('All core modules loaded successfully!~n', []))),

  asserta((test_advanced :-
    format('Loading advanced recursion test suite...~n', []),
    (use_module(unifyweaver(core/advanced/test_advanced)) ->
      (format('~nRunning tests...~n~n', []),
       test_all_advanced)
    ; format('Failed to load advanced test suite~n', [])))),

  asserta((check_paths :-
    forall(file_search_path(unifyweaver, P),
           format('unifyweaver search path: ~w~n', [P])))),

  asserta((help :-
    format('~n=== UnifyWeaver Help ===~n', []),
    format('Core modules:~n', []),
    format('  load_recursive.  - Load the recursive_compiler module~n', []),
    format('  load_stream.     - Load the stream_compiler module~n', []),
    format('  load_template.   - Load the template_system module~n', []),
    format('  load_all_core.   - Load all core modules~n', []),
    format('~nTesting:~n', []),
    format('  test_advanced.   - Run advanced recursion tests~n', []),
    format('~nUtilities:~n', []),
    format('  check_paths.     - Show configured search paths~n', []),
    format('  help.            - Show this help~n~n', [])))
)
PL

exec swipl -q -g "$PROLOG_GOAL" -t prolog
