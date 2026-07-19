:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% FNR and FILENAME as single-input record-identity specials. plawk streams one
% input, so there is no per-file counter reset: FNR is an exact alias for NR
% (the record number), and FILENAME is the input path -- which the native
% driver takes as ARGV[1] ("-" denotes stdin). Both are implemented as
% parser-level aliases (FNR -> special('NR'), FILENAME -> argv_at(1)), so they
% reuse the existing NR / ARGV[N] codegen with no lowering changes.
%
% Scope mirrors what NR / ARGV[N] already support: FNR reads anywhere NR does
% (print, arithmetic, if/while guards, END); FILENAME is a string scalar
% (print, `x = FILENAME`, string-equality via assign-then-compare). FILENAME in
% an END block is a follow-on, same as ARGV[N] in END.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

:- begin_tests(plawk_fnr_filename).

% FNR is the record number, per record: 1, 2, 3 (== NR in the single-input model).
test(fnr_per_record, [condition(clang_available)]) :-
    sdir(Dir),
    build_run_args(Dir, 'fpr', "{ print FNR }\n", "a\nb\nc\n", ['-'], Out),
    assertion(Out == "1\n2\n3\n"), !.

% FNR in a body guard: print only the second record.
test(fnr_guard, [condition(clang_available)]) :-
    sdir(Dir),
    build_run_args(Dir, 'fg', "{ if (FNR == 2) print $0 }\n", "x\ny\nz\n", ['-'], Out),
    assertion(Out == "y\n"), !.

% FNR in arithmetic: FNR + 10 -> 11, 12, 13.
test(fnr_arith, [condition(clang_available)]) :-
    sdir(Dir),
    build_run_args(Dir, 'fa', "{ print FNR + 10 }\n", "a\nb\nc\n", ['-'], Out),
    assertion(Out == "11\n12\n13\n"), !.

% FNR == NR at end of input (a body rule keeps the record counter live for END).
test(fnr_equals_nr_end, [condition(clang_available)]) :-
    sdir(Dir),
    build_run_args(Dir, 'fe', "{ x = FNR } END { print \"end\", NR, FNR }\n",
        "a\nb\nc\n", ['-'], Out),
    assertion(Out == "end 3 3\n"), !.

% FILENAME over stdin ("-"): the driver's ARGV[1] is "-", so FILENAME is "-".
test(filename_stdin, [condition(clang_available)]) :-
    sdir(Dir),
    build_run_args(Dir, 'fs', "{ if (NR == 1) print FILENAME }\n", "x\ny\n", ['-'], Out),
    assertion(Out == "-\n"), !.

% FILENAME reads the input path when a file is given as ARGV[1].
test(filename_from_file, [condition(clang_available)]) :-
    sdir(Dir),
    directory_file_path(Dir, 'ffdata.txt', DataPath),
    setup_call_cleanup(open(DataPath, write, S, [encoding(utf8)]),
        write(S, "one\ntwo\n"), close(S)),
    build_run_args(Dir, 'ff', "{ if (NR == 1) print FILENAME }\n", "", [DataPath], Out),
    format(string(Expect), "~w\n", [DataPath]),
    assertion(Out == Expect), !.

% FILENAME assigned to a scalar then string-compared against the known path.
test(filename_assign_streq, [condition(clang_available)]) :-
    sdir(Dir),
    directory_file_path(Dir, 'fadata.txt', DataPath),
    setup_call_cleanup(open(DataPath, write, S, [encoding(utf8)]),
        write(S, "one\n"), close(S)),
    format(atom(Src),
        "{ f = FILENAME; if (f == \"~w\") print \"match\"; else print \"no\" }\n",
        [DataPath]),
    build_run_args(Dir, 'fas', Src, "", [DataPath], Out),
    assertion(Out == "match\n"), !.

% Identifier boundary: FILENAMEX / FNRX are ordinary identifiers, not the
% specials -- assigning and reading them must not trip the alias.
test(identifier_boundary, [condition(clang_available)]) :-
    sdir(Dir),
    build_run_args(Dir, 'ib',
        "{ FILENAMEX = 5; FNRX = 7; if (NR == 1) print FILENAMEX, FNRX }\n",
        "z\n", ['-'], Out),
    assertion(Out == "5 7\n"), !.

:- end_tests(plawk_fnr_filename).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_fnr_filename', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run_args(Dir, Name, Src, Input, Args, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, Args,
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(0)).
