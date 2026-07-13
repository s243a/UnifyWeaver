:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk row-oriented records (PLAWK_MULTIPASS_CACHE.md §3.6, phase 8.6): the
% ROW CONSTRUCTOR. `TABLE[$k] = row($a, $b, ...)` stores a row built from the
% CHOSEN fields, in that order -- so a writer can project / reorder the input
% columns into a stored row rather than only capturing the whole record
% (`= $0`). The row is the fields joined by the field separator, so a reader's
% field projection recovers them: `records of` by schema name, `rows of` by
% position. Str-value storage, in-run (like `= $0`).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_row_cons).

% `TABLE[$k] = row($a, $b)` parses to set_row_cons/3 with the chosen fields;
% `= $0` still parses to set_row/2.
test(row_cons_parses) :-
    plawk_parse_string(
        "pass { t[$1] = row($3, $2) }\npass rows of t as r { print r[1], r[2] }\n",
        program_passes([],
            [pass([rule(always,
                 [set_row_cons(var(t), field(1), [field(3), field(2)])])]),
             pass_rows(var(r), var(t),
                 [print([assoc(var(r), int(1)), assoc(var(r), int(2))])])],
            [])),
    plawk_parse_string("pass { t[$1] = $0 }\npass rows of t as r { print r[1] }\n",
        program_passes([],
            [pass([rule(always, [set_row(var(t), field(1))])]) | _], [])),
    !.

% row($3, $2) reorders: field 3 then field 2 become the stored row's columns
% 1 and 2. Read positionally with `rows of`. Over "k1 100 alice" ->
% row is "alice 100" -> r[1]=alice, r[2]=100.
test(row_cons_reorder_positional, [condition(clang_available)]) :-
    cdir(Dir),
    Src = "pass { t[$1] = row($3, $2) }\npass rows of t as r { print r[1], r[2] }\n",
    run_sorted(Dir, 'rcp', Src, "k1 100 alice\nk2 250 bob\n", S),
    assertion(S == ["alice 100", "bob 250"]),
    !.

% The same constructed row read by NAME through a schema (cust=col1, amt=col2).
test(row_cons_named_read, [condition(clang_available)]) :-
    cdir(Dir),
    Src = "BEGIN cache(\"$STORE\") { declare rev(name str, amt str) }\n\c
           pass { rev[$1] = row($3, $2) }\n\c
           pass records of rev as r { print r[\"name\"], r[\"amt\"] }\n",
    run_sorted(Dir, 'rcn', Src, "k1 100 alice\nk2 250 bob\n", S),
    assertion(S == ["alice 100", "bob 250"]),
    !.

% A subset / single field: row($2) keeps just field 2.
test(row_cons_subset, [condition(clang_available)]) :-
    cdir(Dir),
    Src = "pass { t[$1] = row($2) }\npass rows of t as r { print r[1] }\n",
    run_sorted(Dir, 'rcs', Src, "a x y\nb p q\n", S),
    assertion(S == ["p", "x"]),
    !.

:- end_tests(plawk_row_cons).

% --- helpers ---------------------------------------------------------------

cdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_row_cons', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

replace_store(Src, Store, Out) :-
    split_string(Src, "", "", [S0]),
    atomic_list_concat(Parts, "$STORE", S0),
    atomic_list_concat(Parts, Store, Out).

run_sorted(Dir, Name, Src0, Input, Sorted) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    replace_store(Src0, Store, Src),
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
