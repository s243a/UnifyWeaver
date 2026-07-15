:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk string-equality guards: `if (s == "text")` / `if (s != "text")` on a
% string scalar. Lowered as an interned atom-id comparison -- interning is
% canonical, so two equal strings share an id and `==`/`!=` reduce to an i64
% icmp of the scalar's atom id against the interned literal. v1 is a single
% comparison (== / != only; ordering and && / || combinations are follow-ons).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_strguard).

% --- parsing ----------------------------------------------------------------

test(eq_guard_parses) :-
    plawk_parse_string("{ if (s == \"x\") print $1 }\n",
        program([], [rule(always,
            [if(scalar_if(cmp(var(s), eq, string("x"))), [print([field(1)])], [])])], [])),
    !.

test(ne_guard_parses) :-
    plawk_parse_string("{ if (s != \"x\") print $1 }\n",
        program([], [rule(always,
            [if(scalar_if(cmp(var(s), ne, string("x"))), [print([field(1)])], [])])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% `==` on a string scalar built by concat: fires only on the matching record.
test(eq_match, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'eq',
        "{ full = $1 $2; if (full == \"abcd\") print \"match\" }\n",
        "ab cd\nx y\n", Out, St),
    assertion(St == 0), assertion(Out == "match\n"), !.

% `!=` filters out the matching value.
test(ne_filter, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ne',
        "{ k = $1 \"\" ; if (k != \"skip\") print $1 }\n",
        "a\nskip\nb\n", Out, St),
    assertion(St == 0), assertion(Out == "a\nb\n"), !.

% `==` on a single-field string scalar, with an else via a second guard.
test(eq_selects, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'sel',
        "{ tag = $1 \"\" ; if (tag == \"err\") print \"E\" }\n",
        "ok\nerr\nerr\nok\n", Out, St),
    assertion(St == 0), assertion(Out == "E\nE\n"), !.

% the guard composes with an else branch (plain if/else).
test(eq_if_else, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ie',
        "{ tag = $1 \"\" ; if (tag == \"y\") print \"yes\"; else print \"no\" }\n",
        "y\nn\n", Out, St),
    assertion(St == 0), assertion(Out == "yes\nno\n"), !.

% --- ordering (< <= > >=, via strcmp) ---------------------------------------

% `<`: select values that sort before a literal.
test(lt_orders_lexically, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'lt', "{ k = $1 \"\" ; if (k < \"m\") print $1 }\n",
        "apple\nzebra\nmango\nbanana\n", Out, St),
    assertion(St == 0), assertion(Out == "apple\nbanana\n"), !.

% `>=`: the complement.
test(ge_orders_lexically, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ge', "{ k = $1 \"\" ; if (k >= \"m\") print $1 }\n",
        "apple\nzebra\nmango\nbanana\n", Out, St),
    assertion(St == 0), assertion(Out == "zebra\nmango\n"), !.

% `>` on a sprintf-built zero-padded key (composes with sprintf).
test(gt_on_built_key, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'gt',
        "{ tag = sprintf(\"%03d\", $1 + 0); if (tag > \"005\") print $1 }\n",
        "3\n9\n1\n7\n", Out, St),
    assertion(St == 0), assertion(Out == "9\n7\n"), !.

:- end_tests(plawk_strguard).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_strguard', Dir),
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
