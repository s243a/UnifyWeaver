:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Reading an ABSENT associative-array element in STRING context. In awk an
% uninitialized array element is a dual value: the empty string in string context
% and 0 in numeric context. plawk tables are i64-valued, so `print arr[k]` on a
% missing key used to print "0" where awk prints nothing.
%
% Every assoc value PRINT now goes through @wam_assoc_i64_print, which probes the
% occupied bit and prints the stored i64 only when the key exists -- so an absent
% element contributes no bytes, while a STORED zero still prints "0" (the probe
% tests presence, not the value). Numeric contexts are unchanged: they keep
% consuming @wam_assoc_i64_get, whose 0 for a missing key is exactly awk's
% numeric reading. Binary (writebin) output also keeps the numeric 0, since a
% fixed-layout binary field needs a number.
%
% DELIBERATE DEVIATION (not changed here): in awk, merely READING arr[k] creates
% the element (autovivification), so a later for-in sees it and counts it. plawk
% reads stay pure -- a read never inserts -- so reading cannot perturb a later
% iteration or count. Membership (`k in arr`) does not autovivify in awk either,
% which plawk already matched.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_absent_key_read).

% --- absent element prints as empty -----------------------------------------

% Rule-body read of a field key that was never counted: awk prints nothing
% between the brackets.
test(absent_field_key_prints_empty, [condition(clang_available)]) :-
    kdir(Dir),
    build_run(Dir, 'bf', "{ c[$1]++ } { print \"[\" c[$2] \"]\" }\n",
        "a b\n", Out),
    assertion(Out == "[]\n"), !.

% The same read for a key that IS present prints its count.
test(present_field_key_prints_value, [condition(clang_available)]) :-
    kdir(Dir),
    build_run(Dir, 'pf', "{ c[$1]++ } { print \"[\" c[$1] \"]\" }\n",
        "a\na\n", Out),
    assertion(Out == "[1]\n[2]\n"), !.

% An END literal-key read of a missing key prints an empty line.
test(absent_end_literal_key, [condition(clang_available)]) :-
    kdir(Dir),
    build_run(Dir, 'el', "{ c[$1]++ } END { print c[\"nope\"] }\n",
        "a\n", Out),
    assertion(Out == "\n"), !.

% ... and a present END literal key still prints its count.
test(present_end_literal_key, [condition(clang_available)]) :-
    kdir(Dir),
    build_run(Dir, 'ep', "{ c[$1]++ } END { print c[\"a\"] }\n",
        "a\na\n", Out),
    assertion(Out == "2\n"), !.

% A multi-dimensional read of an absent tuple prints empty.
test(absent_multi_dim_key, [condition(clang_available)]) :-
    kdir(Dir),
    build_run(Dir, 'md', "{ c[$1,$2]++; print \"[\" c[$2,$1] \"]\" }\n",
        "a b\n", Out),
    assertion(Out == "[]\n"), !.

% An END for-in that looks up a SECOND table by the loop key: keys present in
% both print their count, keys missing from the second print empty (so the line
% ends with the separator and nothing after it).
test(absent_cross_table_in_end_forin, [condition(clang_available)]) :-
    kdir(Dir),
    build_run_sorted(Dir, 'ct',
        "{ a[$1]++ }\n$2 == \"y\" { b[$1]++ }\nEND { for (k in a) print k, a[k], b[k] }\n",
        "p y\nq n\np y\n", Lines),
    assertion(Lines == ["p 2 2", "q 1 "]), !.

% A rule-body for-in cross-table lookup behaves the same way: the one iterated
% key is missing from `b`, so the line is empty.
test(absent_cross_table_in_rule_forin, [condition(clang_available)]) :-
    kdir(Dir),
    build_run(Dir, 'rf', "{ a[$1]++ } { for (k in a) print b[k] }\n",
        "x\n", Out),
    assertion(Out == "\n"), !.

% --- a STORED zero is still present, and prints "0" -------------------------

% `c[$1] += 0` stores a zero. It is a real element, so for-in sees it and it
% prints as "0" -- the presence probe must not confuse it with an absent key.
test(stored_zero_prints_zero, [condition(clang_available)]) :-
    kdir(Dir),
    build_run(Dir, 'sz', "{ c[$1] += 0 } END { for (k in c) print k, c[k] }\n",
        "a\n", Out),
    assertion(Out == "a 0\n"), !.

% A counted-then-deleted key is absent again, so END's for-in does not see it
% (and the surviving key still prints its count).
test(deleted_key_absent_from_forin, [condition(clang_available)]) :-
    kdir(Dir),
    build_run_sorted(Dir, 'dk',
        "{ c[$1]++ }\n$1 == \"rm\" { delete c[$1] }\nEND { for (k in c) print k, c[k] }\n",
        "a\na\nrm\n", Lines),
    assertion(Lines == ["a 2"]), !.

% --- numeric context keeps awk's 0 ------------------------------------------

% In numeric context an absent element is 0, so `c[$2] + 0` prints 0 rather than
% the empty string -- the dual nature of awk's uninitialized value.
test(absent_key_numeric_context_is_zero, [condition(clang_available)]) :-
    kdir(Dir),
    build_run(Dir, 'nc', "{ c[$1]++ } { print c[$2] + 0 }\n",
        "a b\n", Out),
    assertion(Out == "0\n"), !.

% --- reads stay pure (documented deviation from awk) ------------------------

% Reading c[$2] (a key never counted) prints empty. In awk that read would also
% CREATE the element, so the END cardinality would be 2; plawk reads never
% insert, so it stays 1. This pins plawk's documented behaviour and will fail
% loudly if reads ever start autovivifying.
test(read_does_not_autovivify, [condition(clang_available)]) :-
    kdir(Dir),
    build_run(Dir, 'av',
        "{ c[$1]++ } { print c[$2] } END { n = 0; for (k in c) n++; print n }\n",
        "a b\n", Out),
    assertion(Out == "\n1\n"), !.

% --- STR-VALUED (row / split) tables ----------------------------------------
%
% A row capture (`t[$k] = $0`), a row constructor, and `split` pieces store the
% bytes as an interned atom id, so those tables are str-valued and a read must
% resolve the id back to text. Two bugs lived here: the PLAN-level str-array test
% did not recognise row-capture writers (so a row read printed the raw atom id,
% e.g. "98"), and resolving an ABSENT key's id 0 printed either an unrelated
% static atom or "(null)". Both now go through @wam_assoc_str_print.

% A present row read prints the captured record, not its atom id.
test(row_table_present_read_prints_text, [condition(clang_available)]) :-
    kdir(Dir),
    build_run(Dir, 'rp', "{ t[$1] = $0 } END { print t[\"a\"] }\n",
        "a b\nc d\n", Out),
    assertion(Out == "a b\n"), !.

% An absent row read prints empty (not id 0's text, and not "(null)").
test(row_table_absent_read_prints_empty, [condition(clang_available)]) :-
    kdir(Dir),
    build_run(Dir, 'ra', "{ t[$1] = $0 } END { print t[\"nope\"] }\n",
        "a b\n", Out),
    assertion(Out == "\n"), !.

% Present and absent row reads on one line: the absent one contributes nothing
% after the separator.
test(row_table_mixed_reads, [condition(clang_available)]) :-
    kdir(Dir),
    build_run(Dir, 'rm', "{ t[$1] = $0 } END { print t[\"a\"], t[\"zz\"] }\n",
        "a b\n", Out),
    assertion(Out == "a b \n"), !.

% for-in over a row table resolves every stored row (all keys present).
test(row_table_forin_resolves_rows, [condition(clang_available)]) :-
    kdir(Dir),
    build_run_sorted(Dir, 'rf2', "{ t[$1] = $0 } END { for (k in t) print t[k] }\n",
        "a b\nc d\n", Lines),
    assertion(Lines == ["a b", "c d"]), !.

% A `split` array is str-valued and positionally keyed: an out-of-range position
% is absent, so it prints empty rather than "(null)".
test(split_absent_position_prints_empty, [condition(clang_available)]) :-
    kdir(Dir),
    build_run(Dir, 'sa', "{ split($0, a, \",\"); print a[5] }\n",
        "x,y\n", Out),
    assertion(Out == "\n"), !.

% ... and an in-range position still prints its piece.
test(split_present_position_prints_piece, [condition(clang_available)]) :-
    kdir(Dir),
    build_run(Dir, 'sp', "{ split($0, a, \",\"); print a[1] }\n",
        "x,y\n", Out),
    assertion(Out == "x\n"), !.

% An END for-in that looks up a str table by the loop key: keys the str table
% never captured print empty.
test(absent_str_cross_table_in_end_forin, [condition(clang_available)]) :-
    kdir(Dir),
    build_run_sorted(Dir, 'xe',
        "{ c[$1]++ }\n$2 == \"x\" { t[$1] = $0 }\nEND { for (k in c) print k, t[k] }\n",
        "p x\nq n\n", Lines),
    assertion(Lines == ["p p x", "q "]), !.

% The same cross-table str lookup inside a rule-body for-in.
test(absent_str_cross_table_in_rule_forin, [condition(clang_available)]) :-
    kdir(Dir),
    build_run(Dir, 'xr',
        "{ c[$1]++ }\n$2 == \"x\" { t[$1] = $0 }\n{ for (k in c) print t[k] }\n",
        "p x\nq n\n", Out),
    assertion(Out == "p x\np x\n\n"), !.

:- end_tests(plawk_absent_key_read).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

kdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_absent_key_read', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Src, Bin) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)).

build_run(Dir, Name, Src, Input, Out) :-
    write_prog(Dir, Name, Src, Bin),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(0)).

build_run_sorted(Dir, Name, Src, Input, SortedLines) :-
    write_prog(Dir, Name, Src, Bin),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(0)),
    split_string(Out, "\n", "", Parts0),
    exclude(==(""), Parts0, Parts),
    msort(Parts, SortedLines).
