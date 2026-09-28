:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% `$0` copied into a variable, and field keys past NF in a body read.
%
% `{ s = $0; print s }` printed EMPTY lines: the text-slot store and the concat
% builder projected `$0` through @wam_atom_field_slice_value, which is 1-based --
% field 0 is a null slice. So `s = $0`, `s = $0 "|"`, `k = $0; c[k]++` and a body
% read `c[$0]` all saw "" (the last two agreed with each other, so counts came out
% plausible but wrong). $0 now reads the record's own text
% (plawk_field_text_lines/6).
%
% A field-keyed body read past NF (`print c[$3]` on a two-field record) is the key
% "", like the field-key updates; its null slice pointer is swapped for an empty
% constant before the intern instead of interning a null pointer.
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 1\nb 2\n\na 1\nb\n").

:- begin_tests(plawk_record_copy).

test(record_copy_prints_the_record, [condition(clang_available)]) :-
    run("{ s = $0; print s }\n", "a 1\nb 2\n\na 1\nb\n"),
    !,
    run("{ s = $0; t = s; print t }\n", "a 1\nb 2\n\na 1\nb\n"),
    !,
    run("{ s = $0; if (s == \"a 1\") print \"hit\" }\n", "hit\nhit\n"),
    !.

test(record_in_a_concat, [condition(clang_available)]) :-
    run("{ s = $0 \"|\"; print s }\n", "a 1|\nb 2|\n|\na 1|\nb|\n"),
    !,
    run("{ s = $1 \"-\" $0; print s }\n", "a-a 1\nb-b 2\n-\na-a 1\nb-b\n"),
    !.

% the record as an array key: a scalar key and a body read agree, and both are the
% record text (they both read "" before, so the counts ran 1..5)
test(record_as_an_array_key, [condition(clang_available)]) :-
    run("{ k = $0; c[k]++; n++; print c[$0] \"|\" }\n", "1|\n1|\n1|\n2|\n1|\n"),
    !.

test(field_key_read_past_nf_is_the_empty_key, [condition(clang_available)]) :-
    run("{ c[$3]++; n++; print c[$3], n }\n", "1 1\n2 2\n3 3\n4 4\n5 5\n"),
    !,
    run("{ c[$2]++; n++; print c[$2] \"|\" c[$3] }\n", "1|\n1|\n1|1\n2|1\n2|2\n"),
    !.

:- end_tests(plawk_record_copy).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_record_copy', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    odir(Dir),
    directory_file_path(Dir, 'rc_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'rc', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(O)), stderr(null), process(BPid)]),
    read_string(O, _, _), close(O),
    process_wait(BPid, exit(BuildStatus)),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).
