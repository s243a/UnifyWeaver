:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk row-oriented records (PLAWK_MULTIPASS_CACHE.md §3.6): the first ROW
% WRITER. `TABLE[$k] = $0` captures the whole current record ($0) as a row
% value in TABLE, keyed by field k -- the first producer of row-valued tables
% (a str-value: the record's bytes, interned to a stable id, replace
% semantics). A later pass reads the row back: `over TABLE as k` prints the
% key and `TABLE[k]` resolves the stored id to the row's text. This is the
% storage the named-column `records of TABLE as r` reader (next sub-phase)
% builds on. In-run (one process); cross-run durable rows need byte-valued
% cache storage, a follow-on.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_row_capture).

% `TABLE[$k] = $0` parses to set_row/2; the scalar `NAME = expr` is unchanged.
test(set_row_parses) :-
    plawk_parse_string(
        "pass { orders[$1] = $0 }\npass over orders as k { print k, orders[k] }\n",
        program_passes([],
            [pass([rule(always, [set_row(var(orders), field(1))])]),
             pass_over(var(k), var(orders),
                 [print([var(k), assoc(var(orders), var(k))])])],
            [])),
    plawk_parse_string("{ x = 5 }\nEND { print x }\n",
        program(_, [rule(always, [set(var(x), int(5))])], _)),
    !.

% Capture each record keyed by $1, read the stored rows back. Replace
% semantics: a repeated key keeps the LAST record. Over the three lines,
% key a -> "a 20 z" (last a), key b -> "b 5 y".
test(row_capture_readback, [condition(clang_available)]) :-
    rdir(Dir),
    Src = "pass { orders[$1] = $0 }\npass over orders as k { print k, orders[k] }\n",
    run_sorted(Dir, 'row', Src, "a 10 x\nb 5 y\na 20 z\n", S),
    assertion(S == ["a a 20 z", "b b 5 y"]),
    !.

% The stored row is the full record text (all fields), not just the key.
test(row_is_full_record, [condition(clang_available)]) :-
    rdir(Dir),
    Src = "pass { t[$1] = $0 }\npass over t as k { print t[k] }\n",
    run_sorted(Dir, 'full', Src, "k1 alpha beta\nk2 gamma\n", S),
    assertion(S == ["k1 alpha beta", "k2 gamma"]),
    !.

:- end_tests(plawk_row_capture).

% --- helpers ---------------------------------------------------------------

rdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_row_capture', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run_sorted(Dir, Name, Src, Input, Sorted) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], 0),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out), close(PS), process_wait(Pid, exit(0)),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, Sorted).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
