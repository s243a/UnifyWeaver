:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% POSIX strnum duality extended to non-field sources: a scalar assigned from a
% command-line argument (`x = ARGV[N]`) or a getline read (`getline v < "file"`,
% `s = getline v < "file"`) is a strnum -- its runtime content decides numeric
% vs lexical comparison, like a field copy. Previously these were plain string
% scalars, so `if (x > 5)` compared the interned atom id (a bug: every value
% looked "big"); now it dispatches through @wam_strnum_cmp_int on the text.
%
% Safety: a strnum candidate whose reads are unsupported (e.g. used in a concat)
% deactivates and falls back to a plain string scalar -- its prior behaviour --
% so activation never regresses those programs.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

:- begin_tests(plawk_strnum_src).

% ARGV[N] compares numerically when the argument looks numeric: 3 > 5 is false.
test(argv_numeric_small, [condition(clang_available)]) :-
    sdir(Dir),
    build_run_args(Dir, 'as', "{ x = ARGV[2]; if (x > 5) print \"big\"; else print \"small\" }\n",
        "t\n", ['-', '3'], Out),
    assertion(Out == "small\n"), !.

% ARGV[N] numeric big: 100 > 5.
test(argv_numeric_big, [condition(clang_available)]) :-
    sdir(Dir),
    build_run_args(Dir, 'ab', "{ x = ARGV[2]; if (x > 5) print \"big\"; else print \"small\" }\n",
        "t\n", ['-', '100'], Out),
    assertion(Out == "big\n"), !.

% ARGV[N] non-numeric compares lexically: "abc" > "5".
test(argv_lexical, [condition(clang_available)]) :-
    sdir(Dir),
    build_run_args(Dir, 'al', "{ x = ARGV[2]; if (x > 5) print \"big\"; else print \"small\" }\n",
        "t\n", ['-', 'abc'], Out),
    assertion(Out == "big\n"), !.

% string equality against a literal still works on an ARGV strnum.
test(argv_streq, [condition(clang_available)]) :-
    sdir(Dir),
    build_run_args(Dir, 'ae', "{ x = ARGV[2]; if (x == \"10\") print \"eq\"; else print \"ne\" }\n",
        "t\n", ['-', '10'], Out),
    assertion(Out == "eq\n"), !.

% Deactivation safety: an ARGV scalar used in a concat is not a pure strnum, so
% it falls back to a string scalar and the concat still works (no i64 garbage).
test(argv_concat_fallback, [condition(clang_available)]) :-
    sdir(Dir),
    build_run_args(Dir, 'ac', "{ x = ARGV[2]; y = x \"!\"; print y }\n",
        "t\n", ['-', 'hi'], Out),
    assertion(Out == "hi!\n"), !.

% getline var is a strnum: v=7 compares numerically, 7 > 5.
test(getline_numeric, [condition(clang_available)]) :-
    sdir(Dir),
    directory_file_path(Dir, 'gnum.txt', DataPath),
    setup_call_cleanup(open(DataPath, write, S, [encoding(utf8)]),
        write(S, "7\n"), close(S)),
    format(atom(Src), "{ getline v < \"~w\"; if (v > 5) print \"big\"; else print \"small\" }\n",
        [DataPath]),
    build_run_args(Dir, 'gn', Src, "x\n", ['-'], Out),
    assertion(Out == "big\n"), !.

% getline var v=3 -> small (numeric).
test(getline_numeric_small, [condition(clang_available)]) :-
    sdir(Dir),
    directory_file_path(Dir, 'gsm.txt', DataPath),
    setup_call_cleanup(open(DataPath, write, S, [encoding(utf8)]),
        write(S, "3\n"), close(S)),
    format(atom(Src), "{ getline v < \"~w\"; if (v > 5) print \"big\"; else print \"small\" }\n",
        [DataPath]),
    build_run_args(Dir, 'gs', Src, "x\n", ['-'], Out),
    assertion(Out == "small\n"), !.

% capture form `s = getline v < file`: v is a strnum too.
test(getline_capture_strnum, [condition(clang_available)]) :-
    sdir(Dir),
    directory_file_path(Dir, 'gcap.txt', DataPath),
    setup_call_cleanup(open(DataPath, write, S, [encoding(utf8)]),
        write(S, "8\n"), close(S)),
    format(atom(Src), "{ s = getline v < \"~w\"; if (v > 5) print \"big\"; else print \"small\" }\n",
        [DataPath]),
    build_run_args(Dir, 'gc', Src, "x\n", ['-'], Out),
    assertion(Out == "big\n"), !.

:- end_tests(plawk_strnum_src).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_strnum_src', Dir),
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
