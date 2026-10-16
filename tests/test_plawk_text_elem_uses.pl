:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% The rest of array plan item 5(d) (docs/design/PLAWK_ARRAY_VALUE_MODEL.md): text
% elements beyond arithmetic. A bare copy of a strnum element (`x = a[k]`, so the
% max idiom `if (a[k] > m) m = a[k]`) makes x a strnum; an element inside a
% concatenation reads as its text; a STRING-kind element (`a[k] = $2 ""`) compares
% as a string even against a number. All declined.
%
% Also the UNSET comparison operand: an unassigned scalar (or an absent element)
% compares as awk's uninitialised value -- the number 0 against a numeric-looking
% strnum, "" otherwise. It read atom 0 (a NULL), so `x > m` before m was assigned
% printed nothing where gawk prints "x". gawk 5.1.0 (LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 3\nb 10\na 2.5\nc -7\nd x\n").

:- begin_tests(plawk_text_elem_uses).

test(element_copies, [condition(clang_available)]) :-
    run("{ a[$1] = $2; x = a[$1]; print x }\n", "3\n10\n2.5\n-7\nx\n"),
    run("{ a[$1] = $2; x = a[\"zz\"]; print \"[\" x \"]\" }\n", "[]\n[]\n[]\n[]\n[]\n"),
    run("{ a[$1] = $2; if (a[$1] > m) m = a[$1] } END { print m }\n", "x\n"),
    !.

% Before the first assignment m is unset: -3 > unset is numeric (-3 > 0, false).
test(unset_operands, [condition(clang_available)]) :-
    run("{ x = $2; if (x > m) m = x } END { print m }\n", "x\n"),
    run("{ if (u > 5) print \"y\"; else print \"n\"; u = $2 }\n", "n\nn\ny\nn\nn\n"),
    run_on("a -3\n",
        "{ a[$1] = $2 } END { for (k in a) if (a[k] > m) { m = a[k]; mk = k }; print \"[\" mk \"|\" m \"]\" }\n",
        "[|]\n"),
    !.

test(element_in_concatenation, [condition(clang_available)]) :-
    run("{ a[$1] = $2; s = s a[$1] } END { print s }\n", "3102.5-7x\n"),
    run("{ a[$1] = $2; s = $1 \"=\" a[$1] \";\"; print s }\n",
        "a=3;\nb=10;\na=2.5;\nc=-7;\nd=x;\n"),
    !.

test(string_kind_conditions, [condition(clang_available)]) :-
    run("{ a[$1] = $2 \"\"; if (a[$1] > 5) print $1 }\n", "d\n"),
    run("{ a[$1] = $2 \"\"; if (a[$1] < 5.5) print $1 }\n", "a\nb\na\nc\n"),
    run("{ a[$1] = $2 \"\"; if (a[$1]) print $1 }\n", "a\nb\na\nc\nd\n"),
    run("{ a[$1] = $2 \"\"; x = $2; if (a[$1] == x) print $1 }\n", "a\nb\na\nc\nd\n"),
    !.

:- end_tests(plawk_text_elem_uses).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_text_elem_uses', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_on(Input, Src, Expected).

run_on(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'te_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'te', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(path(swipl),
        ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(BuildStatus)),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, Exit),
    assertion(Exit == exit(0)),
    (   Out == Expected
    ->  true
    ;   format(user_error, "~n~w~n  got      ~q~n  expected ~q~n", [Src, Out, Expected]),
        fail
    ).
