:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% smoke_paths.pl - cross-platform writable tmp roots and cleanup for
% the WAM smoke / benchmark test harnesses.
%
% Why this exists
% ---------------
% Many of the build-smoke tests (test_wam_fsharp_dotnet_smoke,
% test_wam_haskell_*_smoke, test_wam_c_target, etc.) hardcoded
% Root = '/tmp' and shelled out to `rm -rf` for cleanup.  Both
% break on native Windows where /tmp does not exist and `rm` is
% not on PATH.
%
% Rather than copy-pasting the same 20-line tmp_root_candidate /
% writable_tmp_root / clean_dir block into every harness, this
% module exposes them once and lets callers say:
%
%   :- use_module('../helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
%
%   smoke_root(Dir) :- tmp_root(Root),
%                      directory_file_path(Root, 'uw_my_smoke', Dir).
%
%   setup :- smoke_root(D), clean_dir(D), make_directory_path(D).
%
% tmp_root/1 precedence (first existing+writable wins)
% ----------------------------------------------------
%   1. UW_SMOKE_TMPDIR        explicit override (per-harness opt-in)
%   2. TMPDIR / TMP / TEMP    standard env vars (Unix / Windows / Cygwin)
%   3. $PREFIX/tmp            Termux convention
%   4. /data/data/com.termux/files/usr/tmp   Termux default path
%   5. /tmp                   Unix default
%   6. ./tmp                  cwd-relative last resort
%
% Each candidate is created via make_directory_path/1 (no-op if it
% already exists) and tested with access_file(_, write); the first
% one that succeeds wins.  This mirrors the pattern already used by
% tests/test_wam_haskell_csr_smoke.pl and the Elixir phase-3 harness,
% with the addition of UW_SMOKE_TMPDIR for tests that want a per-harness
% override.
%
% clean_dir/1
% -----------
% Uses library(filesex):delete_directory_and_contents/1 instead of
% process_create(path(rm), ...) so cleanup works on Windows without
% rm on PATH.  Wrapped in catch/3 so a delete failure (e.g. a file
% still locked by a previous dotnet build) does not abort the test.

:- module(smoke_paths,
          [ tmp_root/1,
            tmp_root_candidate/1,
            clean_dir/1,
            python_cmd/1
          ]).

:- use_module(library(filesex), [make_directory_path/1,
                                  directory_file_path/3,
                                  delete_directory_and_contents/1]).
:- use_module(library(process)).

%! tmp_root_candidate(-Root:atom) is multi.
%
% Enumerate candidate tmp roots in precedence order.  Use tmp_root/1
% for the deterministic, first-writable selection.  Exported so a
% harness can override the chain by asserting its own preferences
% before falling through.
tmp_root_candidate(Root) :-
    member(Var, ['UW_SMOKE_TMPDIR', 'TMPDIR', 'TMP', 'TEMP']),
    getenv(Var, Raw),
    Raw \== '',
    Root = Raw.
tmp_root_candidate(Root) :-
    getenv('PREFIX', Prefix),
    Prefix \== '',
    directory_file_path(Prefix, tmp, Root).
tmp_root_candidate('/data/data/com.termux/files/usr/tmp').
tmp_root_candidate('/tmp').
tmp_root_candidate('./tmp').

%! tmp_root(-Root:atom) is det.
%
% Resolve to the first candidate that exists (or can be created)
% and is writable.  Throws no exception when all candidates fail;
% instead, ./tmp is the unconditional last resort and create+write
% there is virtually always possible.
tmp_root(Root) :-
    tmp_root_candidate(Cand),
    catch(make_directory_path(Cand), _, fail),
    access_file(Cand, write),
    !,
    Root = Cand.

%! clean_dir(+Dir:atom) is det.
%
% Recursively remove Dir if it exists.  Errors are swallowed so a
% locked file (Windows often locks build outputs briefly after a
% process exits) does not turn into a test failure.
clean_dir(Dir) :-
    (   exists_directory(Dir)
    ->  catch(delete_directory_and_contents(Dir), _, true)
    ;   true
    ).

%! python_cmd(-Cmd:atom) is semidet.
%
% Resolve a runnable Python 3 interpreter command for use with
% process_create(path(Cmd), ...).  Tries `python` first, then `python3`.
%
% Windows note: `python` is the real CPython executable, while
% `python3` is typically the Microsoft Store app-execution alias
% that exits 49 without doing anything when not installed from the
% Store.  Most Linux distros (and macOS) accept either, so trying
% `python` first is the Windows-friendly default.
%
% Verifies that `Cmd --version` runs and exits 0 (not just that the
% executable is found on PATH) so the Microsoft Store stub is
% rejected.
%
% Fails if neither runs.  Callers can:
%   (   python_cmd(Py)
%   ->  process_create(path(Py), [Script, ...], [...])
%   ;   skip('python not available')
%   ).
python_cmd(Cmd) :-
    member(Try, [python, python3]),
    catch(
        (   process_create(path(Try), ['--version'],
                [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0))
        ),
        _, fail),
    !,
    Cmd = Try.
