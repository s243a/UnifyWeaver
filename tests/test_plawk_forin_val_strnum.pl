:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Array-element comparison on a STRING-VALUED array (split / assoc(str)) in an
% END for-in filter: `for (k in a) { if (a[k] CMP int) ... }`. A split array
% stores each element as an interned atom id, so a raw i64 icmp on the stored
% value would compare registry positions, not the element -- the previous
% behaviour, and a bug. The value comparison now resolves the element text and
% goes through strnum (@wam_strnum_cmp_int): numeric when the element looks like
% a number, lexical otherwise (POSIX duality). Counter tables (arr[k]++, a
% genuine i64) keep the plain i64 icmp -- see test_plawk_forin_filter.pl.
%
% A string RHS: `==`/`!=` are a canonical atom-id icmp (the literal interned from
% a module-level constant -- equal strings share an id, so no strcmp); the
% ordering ops (`<` `<=` `>` `>=`) resolve the element text and strcmp it against
% the literal, testing the sign with the predicate.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_forin_val_strnum).

% --- parsing ---------------------------------------------------------------

% `for (k in a) { if (a[k] > 2) ... }` parses to a guarded END for-in whose
% guard is forin_val_cmp (the same node the counter filter uses).
test(split_value_guard_parses) :-
    plawk_parse_string(
        "{ split($0, a, \",\") }\n\c
         END { for (k in a) { if (a[k] > 2) print k } }\n",
        program(_, _, [end([for_in(var(k), var(a),
            [if(forin_val_cmp(a, k, gt, 2), [print([var(k)])], [])])])])),
    !.

% `a[k] == "str"` parses to a forin_val_cmp with a str(Text) RHS.
test(split_streq_guard_parses) :-
    plawk_parse_string(
        "{ split($0, a, \",\") }\n\c
         END { for (k in a) { if (a[k] == \"b\") print k } }\n",
        program(_, _, [end([for_in(var(k), var(a),
            [if(forin_val_cmp(a, k, eq, str("b")), [print([var(k)])], [])])])])),
    !.

% --- runtime ---------------------------------------------------------------

% Numeric duality: values 5,1,9 (all numeric); `> 2` keeps 5 and 9, so keys
% 1 and 3. The buggy id-compare would have kept key 2 (value 1, id 2) too.
test(split_gt_numeric, [condition(clang_available)]) :-
    vdir(Dir),
    build_run(Dir, 'gt', "{ split($0, a, \",\") }\n\c
        END { for (k in a) { if (a[k] > 2) print k, a[k] } }\n",
        "5,1,9\n", Out),
    sorted_lines(Out, S),
    assertion(S == ["1 5", "3 9"]), !.

% Numeric equality: values 5,1,9,5; `== 5` keeps keys 1 and 4.
test(split_eq_numeric, [condition(clang_available)]) :-
    vdir(Dir),
    build_run(Dir, 'eq', "{ split($0, a, \",\") }\n\c
        END { for (k in a) { if (a[k] == 5) print k } }\n",
        "5,1,9,5\n", Out),
    sorted_lines(Out, S),
    assertion(S == ["1", "4"]), !.

% Lexical fallback: non-numeric elements compare as strings against the
% snprintf of the integer RHS. Values x,3,z,0 with `> 1`: "x">"1" (lexical) T,
% 3>1 (numeric) T, "z">"1" (lexical) T, 0>1 (numeric) F -> keys 1,2,3.
test(split_gt_lexical_mix, [condition(clang_available)]) :-
    vdir(Dir),
    build_run(Dir, 'lx', "{ split($0, a, \",\") }\n\c
        END { for (k in a) { if (a[k] > 1) print k } }\n",
        "x,3,z,0\n", Out),
    sorted_lines(Out, S),
    assertion(S == ["1", "2", "3"]), !.

% A threshold above every element keeps nothing.
test(split_gt_excludes_all, [condition(clang_available)]) :-
    vdir(Dir),
    build_run(Dir, 'ex', "{ split($0, a, \",\") }\n\c
        END { for (k in a) { if (a[k] > 100) print k } }\n",
        "5,1,9\n", Out),
    assertion(Out == ""), !.

% String equality: values a,b,c,b; `== "b"` keeps keys 2 and 4.
test(split_streq, [condition(clang_available)]) :-
    vdir(Dir),
    build_run(Dir, 'se', "{ split($0, a, \",\") }\n\c
        END { for (k in a) { if (a[k] == \"b\") print k } }\n",
        "a,b,c,b\n", Out),
    sorted_lines(Out, S),
    assertion(S == ["2", "4"]), !.

% String inequality: values a,b,c; `!= \"b\"` keeps keys 1 and 3.
test(split_strne, [condition(clang_available)]) :-
    vdir(Dir),
    build_run(Dir, 'sn', "{ split($0, a, \",\") }\n\c
        END { for (k in a) { if (a[k] != \"b\") print k, a[k] } }\n",
        "a,b,c\n", Out),
    sorted_lines(Out, S),
    assertion(S == ["1 a", "3 c"]), !.

% A multi-character literal equality.
test(split_streq_multichar, [condition(clang_available)]) :-
    vdir(Dir),
    build_run(Dir, 'sm', "{ split($0, a, \",\") }\n\c
        END { for (k in a) { if (a[k] == \"foo\") print k } }\n",
        "foo,bar,foo\n", Out),
    sorted_lines(Out, S),
    assertion(S == ["1", "3"]), !.

% String ordering (`< "m"`): strcmp against the literal. apple,zebra,cat,mango
% -> apple and cat are lexically < "m".
test(split_strlt, [condition(clang_available)]) :-
    vdir(Dir),
    build_run(Dir, 'sl', "{ split($0, a, \",\") }\n\c
        END { for (k in a) { if (a[k] < \"m\") print k, a[k] } }\n",
        "apple,zebra,cat,mango\n", Out),
    sorted_lines(Out, S),
    assertion(S == ["1 apple", "3 cat"]), !.

% String ordering (`>= "m"`): the complement -- zebra and mango.
test(split_strge, [condition(clang_available)]) :-
    vdir(Dir),
    build_run(Dir, 'sg', "{ split($0, a, \",\") }\n\c
        END { for (k in a) { if (a[k] >= \"m\") print k, a[k] } }\n",
        "apple,zebra,cat,mango\n", Out),
    sorted_lines(Out, S),
    assertion(S == ["2 zebra", "4 mango"]), !.

% Regression: a genuine i64 counter table still compares numerically as an
% i64 (the value IS the count, not an atom id). a=3,b=1; `> 1` keeps a.
test(counter_still_i64, [condition(clang_available)]) :-
    vdir(Dir),
    build_run(Dir, 'ct', "{ c[$1]++ }\n\c
        END { for (k in c) { if (c[k] > 1) print k, c[k] } }\n",
        "a\nb\na\na\n", Out),
    sorted_lines(Out, S),
    assertion(S == ["a 3"]), !.

:- end_tests(plawk_forin_val_strnum).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

vdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_forin_val_strnum', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

sorted_lines(Out, Sorted) :-
    split_string(Out, "\n", "", L0),
    exclude(==(""), L0, L),
    msort(L, Sorted).

build_run(Dir, Name, Src, Input, Out) :-
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
    process_wait(RPid, exit(0)).
