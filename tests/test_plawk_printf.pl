:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk `printf` format coverage. The subset grew from %s/%d/%ld to the standard
% conversion prefix `%[flags][width][.precision][length]<conv>`: integer args
% (i64) take d/i/x/X/o/u (with the `l` length modifier) and c (code point ->
% char); float args take f/g/e/F/G/E; string args (%s) take flags/width/
% precision; a record-field slice (%s over len+ptr) takes flags/width and rides
% `.*` for its length (a user precision on a field slice is a clean compile
% error -- the pointer is not null-terminated). The format is rewritten to a C
% printf format driven by each argument's inferred kind.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

:- begin_tests(plawk_printf).

% --- integer width / flags --------------------------------------------------

test(int_width, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'iw', "{ printf \"[%5d]\\n\", NR }\n", "a\n", Out, St),
    assertion(St == 0), assertion(Out == "[    1]\n"), !.

test(int_zero_pad, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'iz', "{ printf \"[%05d]\\n\", NR }\n", "a\n", Out, St),
    assertion(St == 0), assertion(Out == "[00001]\n"), !.

test(int_left_justify, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'il', "{ printf \"[%-5d]\\n\", NR }\n", "a\n", Out, St),
    assertion(St == 0), assertion(Out == "[1    ]\n"), !.

% --- hex / octal ------------------------------------------------------------

test(hex_lower_upper, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'hx', "{ printf \"%x %X\\n\", $1 + 0, $1 + 0 }\n", "255\n", Out, St),
    assertion(St == 0), assertion(Out == "ff FF\n"), !.

test(hex_with_width_and_hash, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'hxw', "{ printf \"%#06x\\n\", $1 + 0 }\n", "255\n", Out, St),
    assertion(St == 0), assertion(Out == "0x00ff\n"), !.

test(octal, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'oc', "{ printf \"%o\\n\", $1 + 0 }\n", "8\n", Out, St),
    assertion(St == 0), assertion(Out == "10\n"), !.

% --- %c (code point -> character) -------------------------------------------

test(char_from_code, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ch', "{ printf \"%c\\n\", NR + 64 }\n", "x\ny\n", Out, St),
    assertion(St == 0), assertion(Out == "A\nB\n"), !.

% --- string width / precision -----------------------------------------------

test(string_literal_width, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'sw', "{ printf \"[%6s]\\n\", \"hi\" }\n", "x\n", Out, St),
    assertion(St == 0), assertion(Out == "[    hi]\n"), !.

test(string_literal_precision, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'sp', "{ printf \"%.2s\\n\", \"hello\" }\n", "x\n", Out, St),
    assertion(St == 0), assertion(Out == "he\n"), !.

test(field_slice_width, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'fw', "{ printf \"[%6s]\\n\", $1 }\n", "hi\n", Out, St),
    assertion(St == 0), assertion(Out == "[    hi]\n"), !.

test(field_slice_left_justify, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'fl', "{ printf \"[%-6s]|\\n\", $1 }\n", "hi\n", Out, St),
    assertion(St == 0), assertion(Out == "[hi    ]|\n"), !.

% A precision on a (non-null-terminated) field slice is a clean compile error.
test(field_slice_precision_rejected, [condition(clang_available)]) :-
    ldir(Dir),
    build_status(Dir, 'fp', "{ printf \"%.2s\\n\", $1 }\n", St),
    assertion(St == 3), !.

% --- float width / precision ------------------------------------------------

test(float_width_precision, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'fwp', "{ printf \"[%8.2f]\\n\", float($1) }\n", "3.14159\n", Out, St),
    assertion(St == 0), assertion(Out == "[    3.14]\n"), !.

% --- unchanged basics -------------------------------------------------------

test(plain_d_and_s_unchanged, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'basic', "{ printf \"%s=%d\\n\", $1, NR }\n", "a\nb\n", Out, St),
    assertion(St == 0), assertion(Out == "a=1\nb=2\n"), !.

test(percent_literal, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'pct', "{ printf \"%d%%\\n\", NR }\n", "a\n", Out, St),
    assertion(St == 0), assertion(Out == "1%\n"), !.

% --- mixed ------------------------------------------------------------------

test(mixed_conversions, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'mix', "{ printf \"%-5s|%05d|%x\\n\", $1, $2 + 0, $2 + 0 }\n",
        "ab 42\n", Out, St),
    assertion(St == 0), assertion(Out == "ab   |00042|2a\n"), !.

:- end_tests(plawk_printf).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_printf', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Src, Bin, Prog) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin).

build_status(Dir, Name, Src, Status) :-
    write_prog(Dir, Name, Src, Bin, Prog),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)).

build_run(Dir, Name, Src, Input, Out, RunStatus) :-
    write_prog(Dir, Name, Src, Bin, Prog),
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
