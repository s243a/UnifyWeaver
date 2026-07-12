:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk multi-pass, phase 2 PR-B: the multi-pass EXECUTION driver. A program
% with 2+ `pass { }` blocks runs the record loop once per pass over the
% (re-opened) input, sharing an assoc table across passes, then END. Each
% pass is emitted as its own function (so the fixed per-record SSA %line and
% loop labels are function-local and never collide); main creates the shared
% table once and threads it to each pass as a parameter, and the END for-in
% reads it back. Reading the same input twice into a shared counter table
% therefore doubles the counts -- the observable proof of two passes.
%
% v1 scope: text mode; one shared assoc table; always-rule pass bodies
% (no break/next); an `END { for (k in arr) print ... }`; a file argument
% (re-opened per pass -- stdin is not re-openable). See
% PLAWK_MULTIPASS_CACHE.md.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_multipass).

% Two passes over the same input into one shared counter table double the
% single-pass counts.
test(two_passes_double, [condition(clang_available)]) :-
    md(Dir),
    Src = "pass { c[$1]++ }\npass { c[$1]++ }\nEND { for (k in c) print k, c[k] }\n",
    build_bin(Dir, 'dbl', Src, Bin, 0),
    write_input(Dir, "a\na\nb\n", In),
    run_sorted(Bin, [In], S),
    assertion(S == ["a 4", "b 2"]),
    !.

% Three passes triple; confirms the driver scales past two and re-reads each
% time.
test(three_passes_triple, [condition(clang_available)]) :-
    md(Dir),
    Src = "pass { c[$1]++ }\npass { c[$1]++ }\npass { c[$1]++ }\n\c
           END { for (k in c) print k, c[k] }\n",
    build_bin(Dir, 'tri', Src, Bin, 0),
    write_input(Dir, "a\na\nb\n", In),
    run_sorted(Bin, [In], S),
    assertion(S == ["a 6", "b 3"]),
    !.

% A single `pass { }` is still normalised to the ordinary single-pass driver
% (not the multi-pass one) and behaves like a bare main.
test(single_pass_unchanged, [condition(clang_available)]) :-
    md(Dir),
    Src = "pass { c[$1]++ }\nEND { for (k in c) print k, c[k] }\n",
    build_bin(Dir, 'one', Src, Bin, 0),
    write_input(Dir, "a\na\nb\n", In),
    run_sorted(Bin, [In], S),
    assertion(S == ["a 2", "b 1"]),
    !.

% v1 supports a single shared table; two distinct tables across passes are
% reported as outside the current multi-pass surface (exit 3), not
% mis-compiled.
test(two_tables_rejected) :-
    md(Dir),
    Src = "pass { a[$1]++ }\npass { b[$1]++ }\nEND { for (k in a) print k, a[k] }\n",
    directory_file_path(Dir, 'twotbl.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    directory_file_path(Dir, 'twotbl_bin', Bin),
    cli([build, Prog, '-o', Bin], 3),
    !.

:- end_tests(plawk_multipass).

% --- helpers ---------------------------------------------------------------

md(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_multipass', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_input(Dir, Text, In) :-
    directory_file_path(Dir, 'in.txt', In),
    setup_call_cleanup(open(In, write, S, [encoding(utf8)]),
        write(S, Text), close(S)).

build_bin(Dir, Name, Src, Bin, ExpectedStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

run_sorted(Bin, Args, Sorted) :-
    process_create(Bin, Args,
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, Sorted).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
