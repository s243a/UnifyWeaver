:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk `sprintf("fmt", args...)` -- format into a string scalar. Composes the
% printf format engine (same %[flags][width][.precision]conv coverage, same
% argument kinds) with string-valued scalars: the args + rewritten format are
% snprintf'd into a buffer and the result is interned into a string scalar
% (`x = sprintf(...)`), resolved to text on read/print. Numeric conversions
% need a numeric arg (`$1 + 0` / `NR`), exactly as `printf` does.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_sprintf).

% --- parsing ----------------------------------------------------------------

test(sprintf_parses) :-
    plawk_parse_string("{ x = sprintf(\"%05d\", $1) }\n",
        program([], [rule(always,
            [set(var(x), sprintf(string("%05d"), [field(1)]))])], [])),
    !.

test(sprintf_no_args_parses) :-
    plawk_parse_string("{ x = sprintf(\"hi\") }\n",
        program([], [rule(always, [set(var(x), sprintf(string("hi"), []))])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% zero-padded integer (numeric arg via + 0), read at END.
test(zero_pad_int, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'zp', "{ x = sprintf(\"%05d\", $1 + 0) }\nEND { print x }\n",
        "42\n7\n", Out, St),
    assertion(St == 0), assertion(Out == "00007\n"), !.

% string conversion of a field.
test(string_conv, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'sc', "{ x = sprintf(\"[%s]\", $1) }\nEND { print x }\n",
        "hi\n", Out, St),
    assertion(St == 0), assertion(Out == "[hi]\n"), !.

% hex conversion with width + flags.
test(hex_conv, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'hx', "{ x = sprintf(\"0x%04x\", $1 + 0) }\nEND { print x }\n",
        "255\n", Out, St),
    assertion(St == 0), assertion(Out == "0x00ff\n"), !.

% mixed %s and %d (field slice + NR); printed per record.
test(mixed_conversions, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'mx', "{ x = sprintf(\"%s=%d\", $1, NR); print x }\n",
        "a\nb\n", Out, St),
    assertion(St == 0), assertion(Out == "a=1\nb=2\n"), !.

% a literal-only format (no args).
test(literal_only, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'li', "{ x = sprintf(\"tag\") }\nEND { print x }\n",
        "z\n", Out, St),
    assertion(St == 0), assertion(Out == "tag\n"), !.

% the built string can be equality-guarded (composes with string guards).
test(sprintf_then_guard, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'sg',
        "{ k = sprintf(\"n%d\", NR); if (k == \"n2\") print $1 }\n",
        "a\nb\nc\n", Out, St),
    assertion(St == 0), assertion(Out == "b\n"), !.

:- end_tests(plawk_sprintf).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_sprintf', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, Input, Out, RunStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).
