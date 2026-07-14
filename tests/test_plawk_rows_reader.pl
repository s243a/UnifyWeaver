:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk row-oriented records (PLAWK_MULTIPASS_CACHE.md §3.6, phase 8.5): the
% POSITIONAL row reader. `pass rows of TABLE as r { print r[N], ... }`
% iterates TABLE's stored rows, addressing columns BY POSITION (`r[1]`,
% 1-indexed = field N of the stored row) rather than by schema name -- the
% raw / schema-less counterpart to `records of ... r["col"]`. No `declare`
% schema is required. (The `unsafe` modifier and an inline check-or-rename
% spec are a follow-on; this is the positional core.)

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_rows_reader).

% `pass rows of T as r { print ... }` parses to pass_rows/3; positional
% fields are assoc(var(r), int(N)).
test(rows_parses) :-
    plawk_parse_string(
        "pass { t[$1] = $0 }\npass rows of t as r { print r[1], r[2] }\n",
        program_passes([],
            [pass([rule(always, [set_row(var(t), field(1))])]),
             pass_rows(var(r), var(t),
                 [print([assoc(var(r), int(1)), assoc(var(r), int(2))])])],
            [])),
    !.

% Positional read, no schema: r[1] = field 1, r[2] = field 2. Reordering
% (`r[2], r[1]`) proves positional resolution. Over a=10, b=5, a=20
% (replace): a -> "20 a", b -> "5 b".
test(rows_positional_read, [condition(clang_available)]) :-
    wdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t as r { print r[2], r[1] }\n",
    run_sorted(Dir, 'rows', Src, "a 10\nb 5\na 20\n", S),
    assertion(S == ["20 a", "5 b"]),
    !.

% A single positional column.
test(rows_single_col, [condition(clang_available)]) :-
    wdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t as r { print r[1] }\n",
    run_sorted(Dir, 'rows1', Src, "k1 a b\nk2 c d\n", S),
    assertion(S == ["k1", "k2"]),
    !.

% Arithmetic over positional columns, in f64: field2 / field3 per row.
% Over "a 10 4" / "b 9 2": 10/4 = 2.5, 9/2 = 4.5.
test(rows_column_arith, [condition(clang_available)]) :-
    wdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t as r { print r[1], r[2] / r[3] }\n",
    run_sorted(Dir, 'rca', Src, "a 10 4\nb 9 2\n", S),
    assertion(S == ["a 2.5", "b 4.5"]),
    !.

% Awk-native `$N` field addressing WITHOUT an `as VAR` binding: `pass rows of
% t { print $1, $2 }`. A stored row is a field-separated record, so $N maps to
% its Nth column directly.
test(rows_anon_field_addressing, [condition(clang_available)]) :-
    wdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t { print $1, $2 }\n",
    run_sorted(Dir, 'anon', Src, "a 10\nb 5\n", S),
    assertion(S == ["a 10", "b 5"]),
    !.

% `$N` parses to pass_rows_anon (no `as`), distinct from the `as VAR` form.
test(rows_anon_parses) :-
    plawk_parse_string(
        "pass { t[$1] = $0 }\npass rows of t { print $1, $2 }\n",
        program_passes([],
            [pass([rule(always, [set_row(var(t), field(1))])]),
             pass_rows_anon(var(t), [print([field(1), field(2)])])],
            [])),
    !.

% Awk-native arithmetic over `$N` (f64): `$2 / 2`.
test(rows_anon_arith, [condition(clang_available)]) :-
    wdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t { print $1, $2 / 2 }\n",
    run_sorted(Dir, 'anar', Src, "a 10\nb 6\n", S),
    assertion(S == ["a 5", "b 3"]),
    !.

:- end_tests(plawk_rows_reader).

% --- helpers ---------------------------------------------------------------

wdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_rows_reader', Dir),
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
